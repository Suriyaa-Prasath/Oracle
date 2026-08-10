"""Retrieval: a natural-language query -> the top-k most relevant chunks.

The query is embedded with the *same* model used at ingestion time — a
mismatch here doesn't error, it silently produces garbage rankings.

Results below `score_threshold` are dropped rather than returned as
"least irrelevant". An empty list is meaningful: it's the signal the agent
graph uses to fall back to tools instead of inventing an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import settings
from src.ingest import get_collection, get_embedder


@dataclass
class RetrievedChunk:
    """One search hit, ready to be cited."""

    text: str
    score: float                      # cosine similarity, 0..1 (higher = closer)
    source: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-readable citation, e.g. `resume.pdf p.2`."""
        name = self.source.split("/")[-1].split("\\")[-1]
        section = self.metadata.get("section")
        if self.page:
            return f"{name} p.{self.page}"
        if section:
            return f"{name} — {section}"
        return name


def embed_query(query: str) -> list[float]:
    """Encode a query with the shared sentence-transformers model."""
    return get_embedder().encode(
        query, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()


def retrieve(
    query: str,
    top_k: int | None = None,
    where: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Return up to `top_k` chunks above the score threshold, best first.

    `where` maps to Chroma's metadata filter, e.g. `{"doc_type": "pdf"}`.
    Returns `[]` when nothing clears the threshold — callers must handle that.
    """
    top_k = top_k or settings.top_k
    threshold = settings.score_threshold if score_threshold is None else score_threshold

    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(settings.fetch_k, collection.count()),
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[RetrievedChunk] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        # Collection uses cosine space: distance = 1 - similarity.
        score = 1.0 - float(distance)
        if score < threshold:
            continue
        page = meta.get("page") or None
        hits.append(
            RetrievedChunk(
                text=text,
                score=round(score, 4),
                source=str(meta.get("source", "unknown")),
                page=int(page) if page else None,
                metadata=meta,
            )
        )

    return hits[:top_k]


def format_context(chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
    """Render chunks into a numbered block the synthesis prompt can cite.

    Numbering is what lets the model write "[1]" and the UI expand it back
    into the original source.
    """
    if not chunks:
        return "(no relevant documents found)"

    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        block = f"[{i}] Source: {chunk.label}\n{chunk.text.strip()}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    return "\n\n---\n\n".join(parts)
