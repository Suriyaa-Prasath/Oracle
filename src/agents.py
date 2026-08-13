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
"tool" — when the request needs something outside the documents: a general fact \
about the world, current events, arithmetic, or scheduling a calendar event. \
Any request to book, schedule, arrange or set up a meeting, call or reminder is \
"tool", however conversationally it is phrased.
"direct" — only greetings, small talk, or questions about this chatbot itself. \
Never for a request that asks for something to be done or looked up.

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
Q: Set up a 30 minute intro call next Tuesday at 2pm -> {{"route": "tool", "reason": "scheduling request", "tool": "create_calendar_event", "tool_input": "30 minute intro call next Tuesday at 2pm"}}
Q: Can we book a chat on Friday morning? -> {{"route": "tool", "reason": "scheduling request", "tool": "create_calendar_event", "tool_input": "chat on Friday morning"}}
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

_WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]

_EVENT_PROMPT = """\
Extract calendar event details from the request.

Today is {today} ({weekday}). Use this calendar — do not calculate dates \
yourself:
{reference}

Assume a 1 hour duration and working hours unless the request says otherwise \
("morning" = 09:00, "afternoon" = 14:00).

Request: {request}

Respond with JSON only:
{{"title": "...", "start": "YYYY-MM-DDTHH:MM", "duration_minutes": 60, \
"location": "", "description": ""}}"""


def _upcoming_weekdays(today):
    """Map each weekday name to its next occurrence strictly after today."""
    from datetime import timedelta

    upcoming = {}
    for offset in range(1, 8):
        day = today + timedelta(days=offset)
        upcoming.setdefault(day.strftime("%A").lower(), day)
    return upcoming


def _extract_event_args(request: str) -> dict[str, Any] | None:
    """Turn a natural-language scheduling request into calendar tool arguments.

    Returns None when the model can't produce a usable title and start time —
    the caller reports that as a tool failure rather than inventing a date,
    since a confidently wrong meeting time is worse than no event at all.

    An 8B model cannot reliably do date arithmetic: asked for "next Tuesday" on
    a Thursday, it returned the following Sunday. So it is never asked to
    calculate. The prompt carries a dated calendar of the coming week, and any
    weekday named in the request is enforced against that table afterwards —
    the model only has to copy a date across, and if it fails to, the code
    corrects it.
    """
    from datetime import date, datetime

    today = date.today()
    upcoming = _upcoming_weekdays(today)
    reference = "\n".join(
        f"  {name.capitalize()} = {day.isoformat()}" for name, day in upcoming.items()
    )

    try:
        response = get_llm("router", json_mode=True).invoke(
            _EVENT_PROMPT.format(
                today=today.isoformat(),
                weekday=today.strftime("%A"),
                reference=reference,
                request=request,
            )
        )
        parsed = _extract_json(str(response.content))
    except Exception:
        return None

    title = str(parsed.get("title") or "").strip()
    start = str(parsed.get("start") or "").strip()
    if not title or not start:
        return None

    # If the request named a weekday, that weekday wins over whatever date the
    # model produced. Deterministic, and it cannot drift.
    lowered = request.lower()
    named = next((day for day in _WEEKDAYS if day in lowered), None)
    if named and named in upcoming:
        try:
            proposed = datetime.fromisoformat(start)
            if proposed.strftime("%A").lower() != named:
                corrected = datetime.combine(upcoming[named], proposed.time())
                start = corrected.isoformat(timespec="minutes")
        except ValueError:
            start = f"{upcoming[named].isoformat()}T09:00"

    args: dict[str, Any] = {"title": title, "start": start}
    try:
        args["duration_minutes"] = int(parsed.get("duration_minutes") or 60)
    except (TypeError, ValueError):
        args["duration_minutes"] = 60
    for field in ("location", "description"):
        value = parsed.get(field)
        if isinstance(value, str) and value.strip():
            args[field] = value.strip()
    return args


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
    elif name == "create_calendar_event":
        # The only tool needing more than one field, so it gets a dedicated
        # extraction call rather than complicating the router's output schema
        # for every other tool.
        extracted = _extract_event_args(query)
        if not extracted:
            call.update(
                {
                    "args": {"raw": query},
                    "ok": False,
                    "result": "Could not work out the event details. Try naming the "
                    "event and a specific date and time.",
                }
            )
            return {"tool_calls": calls, "trace": [*state.get("trace", []), "tool"]}
        args = extracted
    else:
        args.setdefault("query", query)

    result = run_tool(name, **args)
    call.update(
        {
            "args": args,
            "ok": result.ok,
            "result": result.content if result.ok else (result.error or "failed"),
            # Carried through for tools that produce an artifact rather than
            # just text — the calendar tool's .ics reaches the UI this way.
            "metadata": result.metadata,
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

_TOOL_ANSWER_PROMPT = """\
Answer the user using only the tool output below. Be brief and natural.

Do not apologise, do not mention documents or resumes, and do not mention that \
a tool was used — just state the result. If a calendar event was created, say \
what was scheduled and when, and tell the user the file is ready to download.

Copy any date, time or number **exactly** as it appears in the tool output. \
Never reformat, recalculate or restate it in your own words — the tool output \
is authoritative and the user is acting on it.

Conversation so far:
{history}

{tools}Question: {question}

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

    # Some tool results are facts the user will act on, not material to
    # summarise. Asked to restate a scheduled date in its own words, an 8B
    # model got it wrong roughly one run in six — writing "Saturday 15 August"
    # above a calendar file that correctly said Friday 14 August. A confident
    # sentence contradicting the attached file is worse than either being wrong
    # alone, and no amount of prompting made it reliable. Tools that set
    # `verbatim` have their own wording returned untouched, which is also a
    # generation call saved.
    verbatim = next(
        (c for c in tool_calls if c.get("ok") and (c.get("metadata") or {}).get("verbatim")),
        None,
    )
    if verbatim:
        return {"answer": str(verbatim["result"]).strip(), "citations": [], "trace": trace}

    # A tool answered and no documents were needed. The document-grounded
    # prompt below opens by apologising for having no evidence, which reads as
    # a failure even when the tool succeeded — so answer from the tool alone.
    if tool_calls and not chunks:
        try:
            response = get_llm("synthesis").invoke(
                _TOOL_ANSWER_PROMPT.format(
                    history=state.get("history") or "(none)",
                    tools=tool_section,
                    question=question,
                )
            )
            return {"answer": str(response.content).strip(), "citations": [], "trace": trace}
        except Exception as exc:
            return {"answer": _llm_error(exc), "citations": [], "trace": trace}

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
