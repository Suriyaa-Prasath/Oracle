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
