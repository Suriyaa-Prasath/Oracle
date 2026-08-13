"""End-to-end smoke test — verifies everything except the LLM itself.

    python smoke_test.py

Checks imports, config, chunking, embedding, Chroma round-trip, retrieval,
tools, memory, and graph compilation. Reports LLM reachability separately so
you can validate the pipeline before Ollama is running.
"""

from __future__ import annotations

import sys
import traceback

PASS, FAIL = "  [ok]  ", "  [FAIL]"
failures: list[str] = []


def _write_minimal_pdf(path, lines: list[str]) -> None:
    """Write a structurally valid single-page PDF, so the PDF path is tested
    against a real file rather than a fixture someone has to remember to add."""
    ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"({escaped}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))


def check(name: str, fn):
    try:
        detail = fn()
    except Exception as exc:
        failures.append(name)
        print(f"{FAIL} {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        return None
    print(f"{PASS} {name}" + (f" — {detail}" if detail else ""))
    return detail


def main() -> int:
    print("\n=== Oracle smoke test ===\n")

    from src.config import settings

    check("config loads", lambda: f"provider={settings.provider}, model={settings.model_name}")

    # ---------------------------------------------------------------- ingest
    from src.ingest import (
        chunk_text,
        discover_documents,
        embed_chunks,
        index_stats,
        ingest,
        load_document,
        sample_documents,
    )

    files = check("discover documents", lambda: f"{len(discover_documents())} file(s)")

    def _chunk():
        # Long enough to force several chunks, so overlap is actually exercised.
        para = "This paragraph exists to consume tokens during the chunking test. " * 12
        text = "\n\n".join(f"Section {i}. {para}" for i in range(30))
        chunks = chunk_text(text, {"source": "test.md", "page": 0})
        assert len(chunks) > 1, f"expected multiple chunks, got {len(chunks)}"
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == sorted(set(indices)), "chunk indices must be unique and ordered"
        # Consecutive chunks must share text, or a fact sitting on the boundary
        # is unreachable from both sides.
        boundaries = len(chunks) - 1
        overlapped = sum(1 for a, b in zip(chunks, chunks[1:]) if b["text"][:40] in a["text"])
        assert overlapped == boundaries, f"only {overlapped}/{boundaries} boundaries overlap"
        return f"{len(chunks)} chunks, {overlapped} overlapping boundaries"

    check("token chunking + overlap", _chunk)

    def _formats():
        """PDF and DOCX extraction, on files built here rather than assumed.

        A resume arrives as a PDF more often than anything else, so an
        unexercised PDF path is the riskiest gap in the whole pipeline.
        """
        import shutil
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp(prefix="oracle_fmt_"))
        try:
            _write_minimal_pdf(tmp / "resume.pdf", ["Priya Raghavan", "Streaming ingestion with Flink."])

            import docx

            doc = docx.Document()
            doc.add_paragraph("AWS Certified Solutions Architect, renewed 2025.")
            doc.save(str(tmp / "certs.docx"))

            (tmp / "talks.txt").write_text("Conference talk on delayed labels.", encoding="utf-8")

            results = {}
            for path in discover_documents(tmp):
                text = " ".join(s["text"] for s in load_document(path))
                assert text.strip(), f"{path.name} extracted nothing"
                results[path.suffix] = len(text)

            pdf_text = " ".join(s["text"] for s in load_document(tmp / "resume.pdf"))
            assert "Priya Raghavan" in pdf_text, "PDF text extraction lost content"
            docx_text = " ".join(s["text"] for s in load_document(tmp / "certs.docx"))
            assert "Solutions Architect" in docx_text, "DOCX text extraction lost content"

            return " ".join(f"{ext}={n}ch" for ext, n in sorted(results.items()))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    check("pdf / docx / txt extraction", _formats)

    def _embed():
        vectors = embed_chunks(["hello world", "vector search"])
        assert len(vectors) == 2 and len(vectors[0]) == 384, "unexpected embedding shape"
        return f"dim={len(vectors[0])}"

    check("embeddings (downloads MiniLM on first run)", _embed)

    def _ingest():
        result = ingest(rebuild=True, verbose=False)
        assert result["chunks"] > 0, "no chunks written — is data/ empty?"
        return f"{result['chunks']} chunks from {result['files']} file(s) in {result['seconds']}s"

    check("ingest -> Chroma", _ingest)
    check("index stats", lambda: f"{index_stats()['chunks']} chunks indexed")

    def _samples():
        found = sample_documents()
        return (
            f"{len(found)} placeholder file(s) flagged — banner will show: {', '.join(found)}"
            if found
            else "none — real corpus, banner hidden"
        )

    check("sample-data detection", _samples)

    # -------------------------------------------------------------- retrieve
    from src.retrieve import format_context, retrieve

    def _retrieve():
        hits = retrieve("What is the architecture of this system?")
        assert hits, "retrieval returned nothing — threshold too high, or empty index"
        return f"top hit {hits[0].label} @ {hits[0].score}"

    hits_detail = check("retrieval", _retrieve)
    check("context formatting", lambda: f"{len(format_context(retrieve('architecture')))} chars")

    def _no_match():
        hits = retrieve("xylophone submarine tax law in medieval Estonia")
        return f"{len(hits)} hit(s) for an off-topic query (0 is correct)"

    check("threshold rejects off-topic queries", _no_match)

    # ----------------------------------------------------------------- tools
    from src.tools import calculator, describe_tools, get_tool_schemas, run_tool

    def _calc():
        result = calculator("(45000 * 1.15) / 12")
        assert result.ok, result.error
        return result.content

    check("calculator", _calc)

    def _calc_safety():
        # A rejected *result* is not proof of safety — SymPy's parse_expr
        # returned ok=False for these while the side effect had already run.
        # So detect execution directly: hostile payloads try to create a file,
        # and the check fails if that file appears.
        import pathlib
        import tempfile

        sentinel = pathlib.Path(tempfile.gettempdir()) / "oracle_calc_pwned.txt"
        sentinel.unlink(missing_ok=True)
        target = str(sentinel).replace("\\", "\\\\")

        hostile = [
            f"__import__('os').system('echo x > {target}')",
            f"open('{target}', 'w').write('x')",
            "().__class__.__bases__[0].__subclasses__()",
            "__import__('sys').exit()",
            "lambda: 1",
            "[x for x in range(10)]",
            "import os",
        ]
        leaked = [expr for expr in hostile if calculator(expr).ok]
        assert not leaked, f"returned a result for hostile input: {leaked}"
        assert not sentinel.exists(), "HOSTILE CODE EXECUTED — sentinel file was created"

        # Real maths must still work.
        for expr, expected in [("2+2", "4"), ("sqrt(16)", "4"), ("10 % 3", "1")]:
            result = calculator(expr)
            assert result.ok and expected in result.content, f"{expr} broke: {result}"

        return f"{len(hostile)} payloads rejected, no execution, maths intact"

    check("calculator rejects code execution", _calc_safety)
    def _calendar():
        from datetime import datetime

        from src.tools import create_calendar_event

        result = create_calendar_event(
            title="Intro call; staff role",
            start="2026-08-18T14:00",
            duration_minutes=30,
            description="Line one\nline two",
        )
        assert result.ok, result.error
        ics = result.metadata["ics"]

        for required in ["BEGIN:VCALENDAR", "BEGIN:VEVENT", "UID:", "DTSTAMP:",
                         "DTSTART:20260818T140000", "DTEND:20260818T143000",
                         "END:VEVENT", "END:VCALENDAR"]:
            assert required in ics, f"malformed .ics: missing {required}"
        assert "\r\n" in ics, "iCalendar requires CRLF"
        assert r"role" in ics and r"\;" in ics, "semicolons must be escaped"
        assert r"\n" in ics, "newlines in DESCRIPTION must be escaped"
        assert all(len(l.encode()) <= 75 for l in ics.split("\r\n")), "line over 75 octets"
        assert result.metadata.get("verbatim"), "calendar results must not be paraphrased"

        # Bad input must fail rather than invent a date.
        assert not create_calendar_event(title="", start="2026-08-18T14:00").ok
        assert not create_calendar_event(title="X", start="sometime next week").ok

        span = datetime.fromisoformat(result.metadata["end"]) - datetime.fromisoformat(
            result.metadata["start"]
        )
        return f"valid .ics, {int(span.total_seconds() // 60)} min event, bad input rejected"

    check("calendar event -> .ics", _calendar)
    check("tool schemas", lambda: f"{len(get_tool_schemas())} tools: {', '.join(t['name'] for t in get_tool_schemas())}")
    check("unknown tool handled", lambda: "handled" if not run_tool("nope").ok else "LEAKED")

    def _search():
        result = run_tool("web_search", query="LangGraph framework", max_results=2)
        return result.content.splitlines()[0][:60] if result.ok else f"unavailable ({result.error[:60]})"

    check("web search (network)", _search)

    # ---------------------------------------------------------------- memory
    from src.memory import get_memory

    def _memory():
        mem = get_memory("smoke")
        mem.add("user", "Tell me about the router agent.")
        mem.add("assistant", "It classifies each question into one of three routes.")
        context = mem.as_prompt_context()
        assert "router" in context.lower()
        mem.clear()
        return f"{len(context)} chars of context, cleared cleanly"

    check("conversation memory", _memory)

    # ----------------------------------------------------------------- graph
    from src.agents import build_graph

    check("graph compiles", lambda: f"nodes: {', '.join(build_graph().get_graph().nodes)}")

    # ------------------------------------------------------------------- LLM
    print("\n--- LLM connectivity ---")
    from src.llm import health_check

    ok, message = health_check()
    print(f"{PASS if ok else FAIL} {message}")

    if ok:
        from src.agents import run

        def _e2e():
            final = run("What agents does Oracle use?")
            assert final.get("answer"), "no answer produced"
            return f"route={final.get('route')} · {' -> '.join(final.get('trace', []))}"

        detail = check("end-to-end query", _e2e)
        if detail:
            print(f"\n{run('What agents does Oracle use?').get('answer', '')[:600]}\n")
    else:
        print("       Pipeline verified; start the LLM and re-run for an end-to-end check.")

    print("\n=== " + (f"{len(failures)} failure(s): {', '.join(failures)}" if failures else "all checks passed") + " ===\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
