"""Oracle — Streamlit chat interface.

    streamlit run app.py

Shows the agent's work as it happens: which route the router picked, which
nodes ran, which tools fired, and which document chunks were retrieved with
their similarity scores. Watching the machinery is half the point of putting
this on a portfolio.
"""

from __future__ import annotations

import uuid

import streamlit as st

from src.agents import get_graph
from src.config import settings
from src.ingest import index_stats, ingest, sample_documents
from src.llm import health_check
from src.memory import get_memory, resolve_references

st.set_page_config(page_title="Oracle", page_icon="🔮", layout="centered")

NODE_LABELS = {
    "router": "Routing the question",
    "research": "Searching documents",
    "tool": "Calling a tool",
    "synthesis": "Writing the answer",
}


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Starting the agent graph…")
def load_graph():
    """Compile the LangGraph app once per process."""
    return get_graph()


@st.cache_resource(show_spinner="Building the document index…")
def ensure_index() -> dict:
    """Ingest documents if the vector store is empty.

    On a deployed instance the filesystem is ephemeral, so `chroma/` won't
    survive a restart — rebuilding at boot from the committed `data/` folder
    takes seconds for a corpus this size and keeps the app self-healing.
    """
    stats = index_stats()
    if stats["chunks"] == 0:
        ingest(verbose=False)
        stats = index_stats()
    return stats


@st.cache_data(show_spinner=False, ttl=30)
def cached_health() -> tuple[bool, str]:
    return health_check()


# --------------------------------------------------------------------------
# UI pieces
# --------------------------------------------------------------------------

def render_sidebar(stats: dict) -> None:
    """Connection status, index status, and controls."""
    with st.sidebar:
        st.markdown("### 🔮 Oracle")
        st.caption("Multi-agent RAG over resume & portfolio documents")

        ok, message = cached_health()
        (st.success if ok else st.error)(message, icon="✅" if ok else "⚠️")
        if not ok and settings.provider == "ollama":
            st.code("ollama serve", language="bash")

        st.divider()

        st.markdown("**Index**")
        st.metric("Chunks indexed", stats.get("chunks", 0))
        sources = stats.get("sources", [])
        if sources:
            with st.expander(f"{len(sources)} document(s)"):
                for source in sources:
                    st.caption(f"• {source}")
        else:
            st.warning(f"No documents indexed. Add files to `{settings.data_dir.name}/`.")

        if st.button("Re-ingest documents", width="stretch"):
            with st.spinner("Re-indexing…"):
                result = ingest(rebuild=True, verbose=False)
            ensure_index.clear()
            st.success(f"{result['chunks']} chunks from {result['files']} file(s)")
            st.rerun()

        st.divider()

        st.markdown("**Retrieval**")
        settings.top_k = st.slider("Chunks per query (top-k)", 1, 15, settings.top_k)
        settings.score_threshold = st.slider(
            "Similarity threshold", 0.0, 0.9, settings.score_threshold, 0.05,
            help="Chunks below this cosine similarity are discarded. Higher = stricter.",
        )

        st.divider()

        if st.button("New chat", width="stretch"):
            get_memory(st.session_state.session_id).clear()
            st.session_state.transcript = []
            st.rerun()

        st.caption(f"Model: `{settings.model_name}` · Embeddings: `{settings.embed_model}`")


def render_trace(state: dict) -> None:
    """Expander showing the route, node path, tools, and retrieved chunks."""
    route = state.get("route", "?")
    trace = state.get("trace", [])
    chunks = state.get("chunks") or []
    tool_calls = [c for c in (state.get("tool_calls") or []) if "ok" in c]

    summary = f"🧠 {route} · {len(chunks)} chunk(s)"
    if tool_calls:
        summary += f" · {len(tool_calls)} tool call(s)"

    with st.expander(summary):
        st.markdown(f"**Path:** `{' → '.join(trace) or 'n/a'}`")
        if state.get("reasoning"):
            st.caption(f"Router: {state['reasoning']}")

        for call in tool_calls:
            icon = "✅" if call.get("ok") else "⚠️"
            st.markdown(f"{icon} **`{call['name']}`** — `{call.get('args', {})}`")
            st.code(str(call.get("result", ""))[:1200])

        if chunks:
            st.markdown("**Retrieved chunks**")
            for i, chunk in enumerate(chunks, start=1):
                st.markdown(f"`[{i}]` **{chunk.get('label', chunk['source'])}** · score `{chunk['score']}`")
                st.caption(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
        elif route == "research":
            st.caption("No chunks cleared the similarity threshold.")


def answer_question(question: str) -> dict:
    """Stream the graph, updating a live status panel as each node runs."""
    memory = get_memory(st.session_state.session_id)

    with st.status("Thinking…", expanded=True) as status:
        status.write("Resolving the question against conversation history")
        standalone = resolve_references(question, memory)
        if standalone != question:
            status.write(f"Rewrote as: _{standalone}_")

        initial = {
            "question": question,
            "rewritten_question": standalone,
            "history": memory.as_prompt_context(),
            "chunks": [],
            "tool_calls": [],
            "iterations": 0,
            "trace": [],
        }

        final: dict = dict(initial)
        for update in load_graph().stream(initial, stream_mode="updates"):
            for node, patch in update.items():
                status.write(NODE_LABELS.get(node, node))
                final.update(patch or {})

        status.update(label="Done", state="complete", expanded=False)

    return final


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.transcript = []

    stats = ensure_index()
    render_sidebar(stats)

    st.title("🔮 Oracle")
    st.caption(
        "Ask about my experience, projects, and skills. "
        "Answers are grounded in my documents, with the agent's reasoning shown."
    )

    # Loud, unmissable, and shown to every visitor — not a dev-only warning.
    # Sample content answering as though it were a real person is the one
    # failure mode worth being obnoxious about.
    samples = sample_documents()
    if samples:
        st.warning(
            f"**Demo mode — the indexed documents describe a fictional person.** "
            f"Nothing this app says about anyone's experience is real. "
            f"Placeholder files: {', '.join(f'`{s}`' for s in samples)}. "
            f"Replace them in `data/` and re-ingest to make this yours.",
            icon="🎭",
        )

    ok, message = cached_health()
    if not ok:
        st.error(message, icon="⚠️")

    for entry in st.session_state.transcript:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("state"):
                render_trace(entry["state"])

    question = st.chat_input("Ask about my experience, projects, or skills…")
    if not question:
        return

    memory = get_memory(st.session_state.session_id)

    st.session_state.transcript.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        state = answer_question(question)
        answer = state.get("answer", "Something went wrong.")
        st.markdown(answer)
        render_trace(state)

        citations = state.get("citations") or []
        if citations:
            st.caption("Sources: " + " · ".join(f"[{c['n']}] {c['label']}" for c in citations))

    memory.add("user", question)
    memory.add("assistant", answer, citations=citations)
    st.session_state.transcript.append(
        {"role": "assistant", "content": answer, "state": state}
    )


if __name__ == "__main__":
    main()
