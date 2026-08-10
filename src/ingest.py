"""Ingestion pipeline: raw documents in `/data` -> embedded chunks in ChromaDB.

    python -m src.ingest              # ingest everything under data/
    python -m src.ingest --rebuild    # wipe the collection first
    python -m src.ingest --path data/resume.pdf

Chunk IDs are a hash of (source, chunk index), so re-running updates existing
chunks instead of duplicating them.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from src.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

# Files shipped as a working demo carry this marker. The app refuses to present
# them as real, because a portfolio bot confidently reciting a fictional
# person's employment history is the worst failure this project can have.
SAMPLE_MARKER = "SAMPLE-DATA:"


# --------------------------------------------------------------------------
# Shared resources
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder():
    """Load the sentence-transformers model once (it costs seconds each time)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def get_collection():
    """Open (or create) the persistent Chroma collection.

    `hnsw:space=cosine` must match the normalised embeddings we write, and
    cannot be changed after creation — switching embedding models means
    deleting `chroma/` and re-ingesting.
    """
    import chromadb

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


# --------------------------------------------------------------------------
# 1. Discover
# --------------------------------------------------------------------------

def discover_documents(data_dir: Path | None = None) -> list[Path]:
    """Return every supported file under `data_dir`, recursively."""
    data_dir = data_dir or settings.data_dir
    if not data_dir.exists():
        return []
    return sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith(".")
        and p.name.lower() != "readme.md"  # the instructions, not the corpus
    )


# --------------------------------------------------------------------------
# 2. Load
# --------------------------------------------------------------------------

def load_document(path: Path) -> list[dict[str, Any]]:
    """Extract text as `{"text", "metadata"}` segments.

    One segment per page for PDFs, one per top-level heading for Markdown,
    one for the whole file otherwise — so chunking never has to care about
    file formats.
    """
    suffix = path.suffix.lower()
    try:
        # Project-relative paths make for readable citations ("data/resume.pdf").
        rel = str(path.relative_to(settings.data_dir.parent))
    except ValueError:
        # `--path` accepts a file anywhere on disk, which relative_to() rejects
        # outright rather than falling back. Cite it by name in that case.
        rel = path.name
    base = {"source": rel, "filename": path.name, "doc_type": suffix.lstrip(".")}

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        segments = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                segments.append({"text": text, "metadata": {**base, "page": i}})
        return segments

    if suffix == ".docx":
        import docx

        doc = docx.Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"text": text, "metadata": {**base, "page": 0}}] if text.strip() else []

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    if suffix == ".md":
        return _split_markdown_sections(text, base)

    return [{"text": text, "metadata": {**base, "page": 0}}]


def _split_markdown_sections(text: str, base: dict[str, Any]) -> list[dict[str, Any]]:
    """Split Markdown on `#`/`##` headings, keeping the heading as context.

    The heading matters: a chunk reading "Built with Kafka and Redis" is far
    more retrievable when it still carries "## Oracle — RAG system" above it.
    """
    segments: list[dict[str, Any]] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            prefix = f"{current_heading}\n\n" if current_heading else ""
            segments.append(
                {
                    "text": prefix + body,
                    "metadata": {**base, "page": 0, "section": current_heading.lstrip("# ").strip()},
                }
            )

    for line in text.splitlines():
        if line.startswith("# ") or line.startswith("## "):
            flush()
            current_heading = line.strip()
            buffer = []
        else:
            buffer.append(line)
    flush()

    return segments or [{"text": text, "metadata": {**base, "page": 0}}]


# --------------------------------------------------------------------------
# 3. Chunk
# --------------------------------------------------------------------------

def chunk_text(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict[str, Any]]:
    """Split one segment into overlapping, token-sized chunks.

    Packs whole paragraphs until the token budget is hit rather than cutting
    mid-sentence, so a resume bullet is never split in half. Token-based (not
    character-based) so chunk sizes match what the embedder actually sees.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    enc = _encoding()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        if not buffer:
            return
        body = "\n\n".join(buffer)
        chunks.append({"text": body, "metadata": {**metadata, "chunk_index": len(chunks)}})

    for para in paragraphs:
        para_tokens = len(enc.encode(para))

        # A single oversized paragraph: hard-split it on token boundaries.
        if para_tokens > chunk_size:
            flush()
            buffer, buffer_tokens = [], 0
            ids = enc.encode(para)
            step = chunk_size - overlap
            for start in range(0, len(ids), step):
                piece = enc.decode(ids[start : start + chunk_size]).strip()
                if piece:
                    chunks.append(
                        {"text": piece, "metadata": {**metadata, "chunk_index": len(chunks)}}
                    )
            continue

        if buffer_tokens + para_tokens > chunk_size:
            flush()
            # Carry the tail of the previous chunk forward as overlap so a fact
            # spanning a boundary is still retrievable from both sides.
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(buffer):
                prev_tokens = len(enc.encode(prev))
                if tail_tokens + prev_tokens > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += prev_tokens

            if not tail and buffer:
                # The trailing paragraph alone exceeds the overlap budget, so
                # the loop above kept nothing. Carry its last `overlap` tokens
                # anyway — otherwise long-paragraph documents (most prose, and
                # most resumes) silently get no overlap at all.
                ids = enc.encode(buffer[-1])[-overlap:]
                piece = enc.decode(ids).strip()
                if piece:
                    tail, tail_tokens = [piece], len(ids)

            buffer, buffer_tokens = tail, tail_tokens

        buffer.append(para)
        buffer_tokens += para_tokens

    flush()
    return chunks


# --------------------------------------------------------------------------
# 4. Embed
# --------------------------------------------------------------------------

def embed_chunks(texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
    """Encode texts with the shared model, L2-normalised for cosine search."""
    texts = list(texts)
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


# --------------------------------------------------------------------------
# 5. Persist
# --------------------------------------------------------------------------

def _chunk_id(source: str, index: int) -> str:
    return hashlib.sha1(f"{source}:{index}".encode()).hexdigest()[:16]


def upsert_to_chroma(chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
    """Write chunks + vectors to Chroma. Returns the number written."""
    if not chunks:
        return 0

    collection = get_collection()
    now = datetime.now(timezone.utc).isoformat()

    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        meta = dict(chunk["metadata"])
        meta["ingested_at"] = now
        # Chroma metadata values must be scalars.
        meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
        ids.append(_chunk_id(meta.get("source", "?"), meta.get("chunk_index", 0)))
        documents.append(chunk["text"])
        metadatas.append(meta)

    collection.upsert(
        ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
    )
    return len(ids)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def ingest(path: Path | None = None, rebuild: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Run the whole pipeline. Returns a summary dict."""
    started = time.perf_counter()

    if rebuild:
        import chromadb

        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass  # nothing to delete on a first run
        get_collection.cache_clear()

    files = [path] if path else discover_documents()
    if not files:
        return {"files": 0, "chunks": 0, "seconds": 0.0, "sources": []}

    all_chunks: list[dict[str, Any]] = []
    for file in files:
        # chunk_index must be unique per source, so number across the whole file
        # rather than restarting at each page.
        file_chunks: list[dict[str, Any]] = []
        for segment in load_document(file):
            for chunk in chunk_text(segment["text"], segment["metadata"]):
                chunk["metadata"]["chunk_index"] = len(file_chunks)
                file_chunks.append(chunk)
        if verbose:
            print(f"  {file.name}: {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    embeddings = embed_chunks(c["text"] for c in all_chunks)
    written = upsert_to_chroma(all_chunks, embeddings)

    return {
        "files": len(files),
        "chunks": written,
        "seconds": round(time.perf_counter() - started, 2),
        "sources": [f.name for f in files],
    }


def sample_documents(data_dir: Path | None = None) -> list[str]:
    """Return the names of indexed files that are still demo placeholders."""
    found: list[str] = []
    for path in discover_documents(data_dir):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:400]
        except OSError:
            continue
        if SAMPLE_MARKER in head:
            found.append(path.name)
    return found


def index_stats() -> dict[str, Any]:
    """Count what's currently in the collection (for the UI sidebar)."""
    try:
        collection = get_collection()
        count = collection.count()
    except Exception:
        return {"chunks": 0, "sources": []}

    sources: list[str] = []
    if count:
        got = collection.get(include=["metadatas"], limit=min(count, 5000))
        sources = sorted({m.get("source", "?") for m in (got.get("metadatas") or [])})
    return {"chunks": count, "sources": sources}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--path", type=Path, help="Ingest a single file instead of all of data/")
    parser.add_argument("--rebuild", action="store_true", help="Delete the collection first")
    args = parser.parse_args()

    print(f"Embedding model: {settings.embed_model}")
    print(f"Source: {args.path or settings.data_dir}")

    result = ingest(path=args.path, rebuild=args.rebuild)
    if not result["files"]:
        print(f"\nNo supported documents found in {settings.data_dir}.")
        print(f"Add {', '.join(sorted(SUPPORTED_EXTENSIONS))} files and re-run.")
        return 1

    print(
        f"\nIngested {result['chunks']} chunks from {result['files']} file(s) "
        f"in {result['seconds']}s -> {settings.chroma_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
