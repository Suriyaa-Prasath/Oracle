"""Tools the agent can call when the vector store isn't enough.

Every tool returns a `ToolResult` and never raises: network failures, rate
limits, and empty results all come back as `ok=False` with a readable error.
A failed web search should degrade the answer, not crash the graph.

Docstrings here are functional — `get_tool_schemas()` derives the schema the
LLM sees from the signature and docstring, so they can't drift apart.

Tools
-----
`web_search`            DuckDuckGo; no API key needed.
`wikipedia_lookup`      Short summaries, with disambiguation surfaced.
`calculator`            Arithmetic over an AST validated node by node.
`create_calendar_event` Scheduling request -> downloadable .ics file.
"""

from __future__ import annotations

import ast
import inspect
import math
import operator
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_CONTENT_CHARS = 2000


@dataclass
class ToolResult:
    """Uniform return type so the agent handles every tool identically."""

    ok: bool
    content: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def truncated(self, limit: int = MAX_CONTENT_CHARS) -> "ToolResult":
        """Cap content length before it reaches the prompt."""
        if len(self.content) > limit:
            self.content = self.content[:limit].rstrip() + " …[truncated]"
        return self


def web_search(query: str, max_results: int = 5) -> ToolResult:
    """Search the web and return the top results with titles, snippets, and URLs.

    Use for current events, company information, or anything not covered by the
    ingested resume and portfolio documents. Do NOT use for questions about the
    document owner's own experience — those come from the vector store.

    Args:
        query: Natural-language search query.
        max_results: How many results to return (1-10).
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # pre-rename package name
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max(1, min(max_results, 10))))
    except Exception as exc:
        return ToolResult(ok=False, error=f"Web search failed: {exc}")

    if not results:
        return ToolResult(ok=False, error=f"No web results for {query!r}.")

    lines = [
        f"{i}. {r.get('title', 'Untitled')}\n   {r.get('body', '').strip()}\n   {r.get('href', '')}"
        for i, r in enumerate(results, start=1)
    ]
    return ToolResult(
        ok=True,
        content="\n".join(lines),
        metadata={"count": len(results), "urls": [r.get("href", "") for r in results]},
    ).truncated()


# Arithmetic is evaluated by walking an AST we validate node by node.
#
# The obvious implementations are both unsafe. `eval()` on a model-generated
# string is arbitrary code execution outright, and SymPy's `parse_expr` calls
# `eval` internally — `__import__('os').system('...')` runs through it and the
# side effect lands before the result fails to sympify. Verified, not assumed.
#
# Below, only numeric literals, the listed operators, and the whitelisted
# functions can be reached. Attribute access, subscripts, strings, lambdas, and
# comprehensions are rejected at validation, so there is no path to a callable
# that was not explicitly allowed here.

_BINARY_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "degrees": math.degrees, "radians": math.radians,
    "factorial": math.factorial, "gcd": math.gcd,
}

_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

# Caps on operations that are cheap to write and expensive to run — an
# unbounded `9**9**9` would hang the request.
_MAX_EXPONENT = 1000
_MAX_FACTORIAL = 1000


def _eval_node(node: ast.AST) -> Any:
    """Recursively evaluate a validated arithmetic AST."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers are allowed, got {type(node.value).__name__}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent above {_MAX_EXPONENT} refused")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct calls to allowed functions are permitted")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"function {node.func.id!r} is not allowed")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [_eval_node(arg) for arg in node.args]
        if func is math.factorial and args and args[0] > _MAX_FACTORIAL:
            raise ValueError(f"factorial above {_MAX_FACTORIAL} refused")
        return func(*args)

    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ValueError(f"unknown name {node.id!r}")
        return _CONSTANTS[node.id]

    raise ValueError(f"{type(node).__name__} is not allowed in an expression")


def calculator(expression: str) -> ToolResult:
    """Evaluate a mathematical expression and return the numeric result.

    Handles arithmetic, percentages, powers, roots, logarithms, and trigonometry.

    Args:
        expression: A math expression, e.g. "(45000 * 1.15) / 12".
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval_node(tree)
    except SyntaxError:
        return ToolResult(ok=False, error=f"{expression!r} is not a valid expression.")
    except (ValueError, TypeError, ArithmeticError, OverflowError) as exc:
        return ToolResult(ok=False, error=f"Could not evaluate {expression!r}: {exc}")

    if not isinstance(value, (int, float)):
        return ToolResult(ok=False, error=f"{expression!r} did not produce a number.")

    rendered = f"{value:,.6g}" if isinstance(value, float) else f"{value:,}"
    return ToolResult(
        ok=True,
        content=f"{expression} = {rendered}",
        metadata={"expression": expression, "result": rendered},
    )


def wikipedia_lookup(topic: str, sentences: int = 3) -> ToolResult:
    """Fetch a short Wikipedia summary for a topic, plus the article URL.

    Use for definitions of technologies, companies, people, or concepts
    mentioned in a question.

    Args:
        topic: The article title or search term.
        sentences: How many sentences of the summary to return (1-10).
    """
    try:
        import wikipedia

        summary = wikipedia.summary(topic, sentences=max(1, min(sentences, 10)))
        page = wikipedia.page(topic, auto_suggest=False)
        url = page.url
    except Exception as exc:
        # Disambiguation and "page not found" both land here; surface the
        # options when we have them.
        options = getattr(exc, "options", None)
        if options:
            return ToolResult(
                ok=False,
                error=f"{topic!r} is ambiguous. Did you mean: {', '.join(options[:5])}?",
            )
        return ToolResult(ok=False, error=f"No Wikipedia article for {topic!r}: {exc}")

    return ToolResult(
        ok=True, content=f"{summary}\n\nSource: {url}", metadata={"url": url}
    ).truncated()


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
#
# This produces an .ics file for the user to download, rather than writing to a
# calendar account through an API. The app is publicly deployed and takes input
# from anonymous visitors, so a tool holding write credentials to a real
# calendar would let any of them create events in it. A generated file has no
# account access and no side effects until someone chooses to import it.

_ICS_ESCAPES = {"\\": "\\\\", ";": r"\;", ",": r"\,", "\n": r"\n"}


def _ics_escape(value: str) -> str:
    for char, replacement in _ICS_ESCAPES.items():
        value = value.replace(char, replacement)
    return value


def _ics_fold(line: str) -> str:
    """Fold to 75 octets per RFC 5545; unfolded lines break strict parsers."""
    encoded = line.encode("utf-8")
    if len(encoded) <= 75:
        return line
    parts, start = [], 0
    while start < len(encoded):
        end = min(start + (75 if not parts else 74), len(encoded))
        # Don't split a multi-byte character across the fold.
        while end > start and (encoded[end - 1] & 0xC0) == 0x80:
            end -= 1
        chunk = encoded[start:end].decode("utf-8")
        parts.append(chunk if not parts else " " + chunk)
        start = end
    return "\r\n".join(parts)


def create_calendar_event(
    title: str,
    start: str,
    duration_minutes: int = 60,
    location: str = "",
    description: str = "",
) -> ToolResult:
    """Create a calendar event and return it as a downloadable .ics file.

    Use when the user asks to schedule, book, or set up a meeting, call, or
    reminder. Produces a file the user imports; it does not write to any
    calendar account.

    Args:
        title: Event name, e.g. "Interview with Priya".
        start: Start time in ISO 8601 format, e.g. "2026-08-14T15:00".
        duration_minutes: Event length in minutes.
        location: Optional location or meeting link.
        description: Optional longer notes.
    """
    from datetime import datetime, timedelta, timezone

    if not title.strip():
        return ToolResult(ok=False, error="An event needs a title.")

    try:
        begins = datetime.fromisoformat(start.strip().replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ToolResult(
            ok=False,
            error=f"Could not read {start!r} as a date and time. "
            "Use ISO 8601, e.g. 2026-08-14T15:00.",
        )

    try:
        minutes = int(duration_minutes)
    except (TypeError, ValueError):
        minutes = 60
    minutes = max(1, min(minutes, 60 * 24))

    ends = begins + timedelta(minutes=minutes)
    stamp = "%Y%m%dT%H%M%S"
    # Times are written without a timezone, which iCalendar reads as floating
    # local time — the event lands at the stated wall-clock time wherever it is
    # imported. That matches what someone asking for "3pm Tuesday" expects.
    uid = f"{uuid.uuid4()}@oracle"

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Oracle//Multi-agent RAG//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{datetime.now(timezone.utc).strftime(stamp)}Z",
        f"DTSTART:{begins.strftime(stamp)}",
        f"DTEND:{ends.strftime(stamp)}",
        f"SUMMARY:{_ics_escape(title.strip())}",
    ]
    if location.strip():
        lines.append(f"LOCATION:{_ics_escape(location.strip())}")
    if description.strip():
        lines.append(f"DESCRIPTION:{_ics_escape(description.strip())}")
    lines += ["END:VEVENT", "END:VCALENDAR"]

    ics = "\r\n".join(_ics_fold(line) for line in lines) + "\r\n"
    when = begins.strftime("%A %d %B %Y at %H:%M")
    summary = f"Created “{title.strip()}” for {when} ({minutes} min)"
    if location.strip():
        summary += f", at {location.strip()}"

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "event"
    return ToolResult(
        ok=True,
        content=f"{summary}. The .ics file is ready to download.",
        metadata={
            "ics": ics,
            "filename": f"{slug[:40]}.ics",
            "title": title.strip(),
            "start": begins.isoformat(),
            "end": ends.isoformat(),
            "duration_minutes": minutes,
            # Report this result word for word; do not paraphrase it through
            # the model. See the note on _VERBATIM in src/agents.py.
            "verbatim": True,
        },
    )


TOOL_REGISTRY: dict[str, Callable[..., ToolResult]] = {
    "web_search": web_search,
    "calculator": calculator,
    "wikipedia_lookup": wikipedia_lookup,
    "create_calendar_event": create_calendar_event,
}


def run_tool(name: str, **kwargs: Any) -> ToolResult:
    """Dispatch by name, converting unknown tools and bad args into results."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return ToolResult(
            ok=False,
            error=f"Unknown tool {name!r}. Available: {', '.join(TOOL_REGISTRY)}",
        )
    try:
        return tool(**kwargs)
    except TypeError as exc:
        return ToolResult(ok=False, error=f"Bad arguments for {name!r}: {exc}")


def get_tool_schemas() -> list[dict[str, Any]]:
    """Build JSON tool schemas from each function's signature and docstring."""
    schemas: list[dict[str, Any]] = []

    for name, func in TOOL_REGISTRY.items():
        signature = inspect.signature(func)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in signature.parameters.items():
            annotation = param.annotation
            json_type = "integer" if annotation is int else "string"
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                properties[param_name]["default"] = param.default

        summary = (inspect.getdoc(func) or "").split("\n\n")[0]
        schemas.append(
            {
                "name": name,
                "description": summary,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )

    return schemas


def describe_tools() -> str:
    """One-line-per-tool description for the router/tool-selection prompt."""
    return "\n".join(
        f"- {s['name']}({', '.join(s['parameters']['properties'])}): {s['description']}"
        for s in get_tool_schemas()
    )
