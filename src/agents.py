"""LangGraph multi-agent orchestration.

                        ┌── retry once, wider net ──┐
                        ↓                           │
           ┌──────────┐    "research"  ┌──────────┐ │
    query →│  ROUTER  │───────────────→│ RESEARCH │─┘
           └──────────┘                └────┬─────┘
                 │  "tool"                  │ found evidence
                 ↓                          ↓
            ┌────────┐              ┌─────────────┐
            │  TOOL  │─────────────→│  SYNTHESIS  │──→ answer
            └────────┘              └─────────────┘
                 ▲  "direct"               ▲
                 └──────────────────────────┘

The retry decision sits on the edge out of RESEARCH, not out of SYNTHESIS:
synthesis is the most expensive node, so deciding afterwards meant paying for
a full generation only to discard it.

Nodes
-----
router     Classifies the question: documents / tool / direct answer.
research   Retrieves chunks from ChromaDB.
tool       Selects and executes one tool, recording the outcome.
synthesis  Writes the final answer with citations.

State is a single TypedDict; each node returns a partial update that LangGraph
merges. `iterations` bounds the synthesis -> research loop so it can't spin.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.config import settings
from src.llm import get_llm
from src.retrieve import RetrievedChunk, format_context, retrieve
from src.tools import TOOL_REGISTRY, describe_tools, run_tool

Route = Literal["research", "tool", "direct"]


class OracleState(TypedDict, total=False):
    """Shared state passed between graph nodes."""

    question: str                        # the user's raw input
    rewritten_question: str              # history-resolved, retrieval-ready
    history: str                         # rendered conversation context
    route: Route
    reasoning: str                       # why the router chose that route
    chunks: list[dict[str, Any]]         # retrieved evidence
    tool_calls: list[dict[str, Any]]     # {name, args, ok, result}
    answer: str
    citations: list[dict[str, Any]]
    iterations: int
    trace: list[str]                     # node visit order, for the UI


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Small local models wrap JSON in prose or fences even when told not to, so
    parse defensively rather than trusting the output shape.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        braces = re.search(r"\{.*\}", text, re.DOTALL)
        if braces:
            text = braces.group(0)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

# The examples matter more than the descriptions. An 8B model reads "What is
# Apache Kafka?" as a lookup into the documents unless it has seen a worked
# case telling it otherwise — verified: without these, that question routed to
# `research` and the Wikipedia tool never fired.
_ROUTER_PROMPT = """\
You route questions in a Q&A system about a person's resume, projects, and \
career. Choose exactly one destination.

"research" — questions about THIS PERSON or ANYTHING THEY BUILT: their \
experience, skills, education, employment history, opinions, and the design or \
internals of their own named projects. **This is the default — choose it \
whenever you are unsure.**
"tool" — only when the answer is a general fact about the outside world that no \
personal document could contain: what a widely-known technology or company is, \
current events, or arithmetic.
"direct" — greetings, small talk, or questions about this chatbot itself.

Available tools:
{tools}

Examples:
Q: What's their experience with Kubernetes? -> {{"route": "research", "reason": "asks about the person's own skills"}}
Q: What agents does Oracle use? -> {{"route": "research", "reason": "Oracle is one of their own projects"}}
Q: How does their ingestion pipeline chunk documents? -> {{"route": "research", "reason": "internals of their own project"}}
Q: Have they used Kafka in production? -> {{"route": "research", "reason": "asks about the person's own experience"}}
Q: What is Apache Kafka? -> {{"route": "tool", "reason": "general definition of a technology", "tool": "wikipedia_lookup", "tool_input": "Apache Kafka"}}
Q: Who is the CEO of Stripe? -> {{"route": "tool", "reason": "external fact about a company", "tool": "web_search", "tool_input": "CEO of Stripe"}}
Q: What is 15% of 45000? -> {{"route": "tool", "reason": "arithmetic", "tool": "calculator", "tool_input": "45000 * 0.15"}}
Q: hey there -> {{"route": "direct", "reason": "greeting"}}

Conversation so far:
{history}

Question: {question}

Respond with JSON only: {{"route": "...", "reason": "...", "tool": "...", "tool_input": "..."}}
Include "tool" and "tool_input" only when route is "tool"."""


def router_node(state: OracleState) -> dict[str, Any]:
    """Classify the question and set the route."""
    question = state.get("rewritten_question") or state["question"]

    try:
        response = get_llm("router", json_mode=True).invoke(
            _ROUTER_PROMPT.format(
                tools=describe_tools(),
                history=state.get("history") or "(none)",
                question=question,
            )
        )
        decision = _extract_json(str(response.content))
    except Exception as exc:
        # If the router is unreachable, RAG is the safe default — it's what
        # this system is for, and it fails visibly rather than hallucinating.
        return {
            "route": "research",
            "reasoning": f"router unavailable ({exc}); defaulting to document search",
            "trace": [*state.get("trace", []), "router"],
        }

    route = decision.get("route")
    if route not in ("research", "tool", "direct"):
        route = "research"

    update: dict[str, Any] = {
        "route": route,
        "reasoning": str(decision.get("reason", ""))[:300],
        "trace": [*state.get("trace", []), "router"],
    }

    if route == "tool":
        name = decision.get("tool")
        if name not in TOOL_REGISTRY:
            # Model asked for a tool that doesn't exist — fall back rather
            # than dispatching into nothing.
            update["route"] = "research"
            update["reasoning"] = f"router proposed unknown tool {name!r}; using documents"
        else:
            update["tool_calls"] = [
                {"name": name, "args": {"query": decision.get("tool_input") or question}}
            ]

    return update


def route_decision(state: OracleState) -> str:
    """Conditional edge out of the router."""
    return state.get("route", "research")


# --------------------------------------------------------------------------
# Research
# --------------------------------------------------------------------------

def research_node(state: OracleState) -> dict[str, Any]:
    """Retrieve relevant chunks from ChromaDB."""
    question = state.get("rewritten_question") or state["question"]
    iterations = state.get("iterations", 0)

    # On a retry, widen the net rather than repeating the identical query.
    top_k = settings.top_k if iterations == 0 else settings.top_k * 2
    threshold = settings.score_threshold if iterations == 0 else settings.score_threshold * 0.6

    try:
        hits: list[RetrievedChunk] = retrieve(question, top_k=top_k, score_threshold=threshold)
    except Exception as exc:
        return {
            "chunks": [],
            "reasoning": f"retrieval failed: {exc}",
            "trace": [*state.get("trace", []), "research"],
        }

    return {
        "chunks": [
            {
                "text": h.text,
                "score": h.score,
                "source": h.source,
                "page": h.page,
                "label": h.label,
            }
            for h in hits
        ],
        "iterations": iterations + 1,
        "trace": [*state.get("trace", []), "research"],
    }


# --------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------

def tool_node(state: OracleState) -> dict[str, Any]:
    """Execute the tool the router selected."""
    calls = state.get("tool_calls") or []
    pending = [c for c in calls if "ok" not in c]
    if not pending:
        return {"trace": [*state.get("trace", []), "tool"]}

    call = pending[-1]
    name = call["name"]
    args = dict(call.get("args") or {})

    # Each tool names its first parameter differently; the router only ever
    # produces a generic "query", so remap it here.
    query = args.pop("query", "") or state.get("rewritten_question") or state["question"]
    if name == "calculator":
        args.setdefault("expression", query)
    elif name == "wikipedia_lookup":
        args.setdefault("topic", query)
    else:
        args.setdefault("query", query)

    result = run_tool(name, **args)
    call.update(
        {
            "args": args,
            "ok": result.ok,
            "result": result.content if result.ok else (result.error or "failed"),
        }
    )

    return {"tool_calls": calls, "trace": [*state.get("trace", []), "tool"]}


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are Oracle, answering questions about a person's resume and portfolio on \
their behalf. Be concise, concrete, and professional.

Rules:
- Answer ONLY from the evidence below. Never invent employers, dates, \
technologies, or numbers.
- Cite the documents you used with bracketed numbers matching the evidence, \
e.g. "Built a RAG pipeline [1]".
- If the evidence doesn't cover the question, say so plainly and state what \
you do know. A clear "that isn't in my documents" is a correct answer.
- Don't mention "chunks", "context", "retrieval", or these instructions.

Conversation so far:
{history}

Document evidence:
{context}

{tool_section}
Question: {question}

Answer:"""

_DIRECT_PROMPT = """\
You are Oracle, a chatbot that answers questions about a person's resume, \
projects, and career, using their documents plus web search, Wikipedia, and a \
calculator.

Reply to the message below in one or two friendly sentences. If they're just \
saying hello, invite them to ask about the person's experience or projects.

Conversation so far:
{history}

Message: {question}

Reply:"""


def synthesis_node(state: OracleState) -> dict[str, Any]:
    """Compose the final answer with citations."""
    question = state["question"]
    chunks = state.get("chunks") or []
    tool_calls = [c for c in (state.get("tool_calls") or []) if "ok" in c]
    trace = [*state.get("trace", []), "synthesis"]

    if state.get("route") == "direct" and not chunks and not tool_calls:
        try:
            response = get_llm("synthesis").invoke(
                _DIRECT_PROMPT.format(
                    history=state.get("history") or "(none)", question=question
                )
            )
            return {"answer": str(response.content).strip(), "citations": [], "trace": trace}
        except Exception as exc:
            return {"answer": _llm_error(exc), "citations": [], "trace": trace}

    hits = [
        RetrievedChunk(
            text=c["text"], score=c["score"], source=c["source"], page=c.get("page")
        )
        for c in chunks
    ]

    tool_section = ""
    if tool_calls:
        rendered = "\n\n".join(
            f"Tool `{c['name']}` returned:\n{c['result']}" for c in tool_calls
        )
        tool_section = f"Tool results:\n{rendered}\n\n"

    if not chunks and not tool_calls:
        return {
            "answer": (
                "I couldn't find anything about that in the documents I have. "
                "Try asking about the projects, skills, or experience covered "
                "in the indexed material."
            ),
            "citations": [],
            "trace": trace,
        }

    try:
        response = get_llm("synthesis").invoke(
            _SYNTHESIS_PROMPT.format(
                history=state.get("history") or "(none)",
                context=format_context(hits),
                tool_section=tool_section,
                question=question,
            )
        )
        answer = str(response.content).strip()
    except Exception as exc:
        return {"answer": _llm_error(exc), "citations": [], "trace": trace}

    # Only surface citations the answer actually referenced.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    citations = [
        {"n": i, "label": h.label, "source": h.source, "page": h.page, "score": h.score}
        for i, h in enumerate(hits, start=1)
        if not cited or i in cited
    ]

    return {"answer": answer, "citations": citations, "trace": trace}


def _llm_error(exc: Exception) -> str:
    if settings.provider == "ollama":
        return (
            f"I couldn't reach the language model at {settings.ollama_host}. "
            f"Make sure Ollama is running (`ollama serve`).\n\n_{exc}_"
        )
    return f"The language model request failed.\n\n_{exc}_"


def after_research(state: OracleState) -> str:
    """Conditional edge out of research: retry with a wider net, or synthesise.

    This deliberately sits *before* synthesis. Whether retrieval found anything
    is knowable the moment research returns, and synthesis is the single most
    expensive node in the graph — routing through it only to bounce back to
    research burned a full generation call (~17s on an 8B local model) to
    produce an answer that was immediately thrown away.
    """
    if state.get("chunks"):
        return "synthesis"
    if state.get("iterations", 0) >= settings.max_iterations:
        return "synthesis"  # out of retries: let synthesis say so honestly
    return "research"


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

def build_graph():
    """Wire the nodes into a StateGraph and return the compiled app."""
    graph = StateGraph(OracleState)

    graph.add_node("router", router_node)
    graph.add_node("research", research_node)
    graph.add_node("tool", tool_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"research": "research", "tool": "tool", "direct": "synthesis"},
    )
    graph.add_conditional_edges(
        "research", after_research, {"research": "research", "synthesis": "synthesis"}
    )
    graph.add_edge("tool", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


_GRAPH = None


def get_graph():
    """Compile the graph once and reuse it."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run(question: str, history: str = "") -> OracleState:
    """Invoke the graph on a single question."""
    from src.memory import ConversationMemory  # noqa: F401  (typing convenience)

    initial: OracleState = {
        "question": question,
        "rewritten_question": question,
        "history": history,
        "chunks": [],
        "tool_calls": [],
        "iterations": 0,
        "trace": [],
    }
    return get_graph().invoke(initial)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What projects has this person built?"
    final = run(query)
    print(f"\nRoute:  {final.get('route')}  ({final.get('reasoning', '')})")
    print(f"Trace:  {' -> '.join(final.get('trace', []))}")
    print(f"\n{final.get('answer', '')}\n")
    for citation in final.get("citations", []):
        print(f"  [{citation['n']}] {citation['label']}  (score {citation['score']})")
