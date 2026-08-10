"""Conversation memory for multi-turn dialogue.

Retrieval is stateless: "what about that project?" embeds to nothing useful.
This module supplies the context that turns a follow-up into a standalone,
retrievable question, and keeps the synthesis agent from repeating itself.

Strategy: keep the last `max_history_turns` turns verbatim; once the window
overflows, compress the evicted turns into a running summary with one cheap
LLM call. Sessions are keyed by `session_id` (Streamlit gives one per browser
tab) so two visitors to the portfolio never share history.

Persistence is in-process — it dies with the app, which is fine for a demo.
See the note at the bottom for the durable option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.config import settings


@dataclass
class Turn:
    """A single exchange."""

    role: str                                    # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    citations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationMemory:
    """Rolling window plus a running summary, for one session."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    # Index into `turns` of the last message already folded into `summary`.
    summarised_upto: int = 0

    def add(self, role: str, content: str, **metadata: Any) -> Turn:
        """Append a turn, summarising older ones once the window overflows."""
        turn = Turn(
            role=role,
            content=content,
            citations=metadata.pop("citations", []) or [],
            metadata=metadata,
        )
        self.turns.append(turn)

        # Keep the full transcript for the UI; only the *prompt* window rolls.
        limit = settings.max_history_turns * 2  # a turn is user + assistant
        if len(self.turns) > limit:
            evicted = self.turns[: len(self.turns) - limit]
            new = evicted[self.summarised_upto :]
            if new:
                self.summary = summarise(new, previous=self.summary)
                self.summarised_upto = len(evicted)

        return turn

    def recent(self, n: int | None = None) -> list[dict[str, str]]:
        """Return the last `n` turns as `{"role", "content"}` dicts."""
        n = n or settings.max_history_turns * 2
        return [t.as_dict() for t in self.turns[-n:]]

    def as_prompt_context(self, max_chars: int = 2500) -> str:
        """Render summary + recent turns into a prompt-ready block."""
        parts: list[str] = []
        if self.summary:
            parts.append(f"Earlier in this conversation: {self.summary}")

        for turn in self.recent():
            speaker = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{speaker}: {turn['content']}")

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = "…" + text[-max_chars:]
        return text

    def clear(self) -> None:
        """Reset the session (the UI's "New chat" button)."""
        self.turns.clear()
        self.summary = ""
        self.summarised_upto = 0

    @property
    def is_empty(self) -> bool:
        return not self.turns


# --------------------------------------------------------------------------
# LLM-backed helpers
# --------------------------------------------------------------------------

_REWRITE_PROMPT = """\
Rewrite the user's latest message as a standalone question that makes sense \
without the conversation history. Resolve pronouns and references \
("that project", "there", "it") into explicit names.

If the message is already standalone, return it unchanged. Return ONLY the \
rewritten question, with no preamble.

Conversation so far:
{history}

Latest message: {question}

Standalone question:"""


def _best_score(question: str) -> float:
    """Top retrieval score for a query, or 0.0 if nothing clears the floor."""
    from src.retrieve import retrieve

    hits = retrieve(question, top_k=1)
    return hits[0].score if hits else 0.0


def resolve_references(question: str, memory: ConversationMemory) -> str:
    """Rewrite a follow-up into a standalone, retrievable question.

    "Did he use Docker there?" + history -> "Did Suriya use Docker on the
    Oracle project?" — something the retriever can actually embed. Returns the
    input unchanged on any failure; a bad rewrite is worse than no rewrite.

    Two guards, both learned the hard way. Rewriting costs a full LLM call —
    the single largest fixed cost per turn after synthesis — and when the
    previous turn is unrelated to the current question, the model happily folds
    that stale context in and produces a query that retrieves *nothing*. So we
    skip the call when the question already retrieves well on its own, and
    discard any rewrite that retrieves worse than what the user actually typed.
    Both checks are embedding lookups: milliseconds against seconds.
    """
    if memory.is_empty or len(question.split()) > 40:
        return question

    try:
        baseline = _best_score(question)
    except Exception:
        baseline = 0.0

    # Already standalone — a rewrite can only cost latency or do harm.
    if baseline >= settings.score_threshold + 0.05:
        return question

    try:
        from src.llm import get_llm

        response = get_llm("router").invoke(
            _REWRITE_PROMPT.format(
                history=memory.as_prompt_context(max_chars=1200), question=question
            )
        )
        rewritten = str(response.content).strip().strip('"')
    except Exception:
        return question

    # Guard against a model that ignored the instruction and wrote an essay.
    if not rewritten or len(rewritten) > 4 * len(question) + 120:
        return question

    try:
        if _best_score(rewritten) < baseline - 0.02:
            return question
    except Exception:
        pass

    return rewritten


_SUMMARY_PROMPT = """\
Compress this conversation excerpt into 2-3 sentences capturing the topics \
discussed and any facts established. Write plainly, no preamble.

{previous}{transcript}

Summary:"""


def summarise(turns: list[Turn], previous: str = "") -> str:
    """Compress evicted turns into a few sentences of running summary."""
    if not turns:
        return previous

    transcript = "\n".join(
        f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}" for t in turns
    )
    prefix = f"Existing summary: {previous}\n\nNew messages:\n" if previous else ""

    try:
        from src.llm import get_llm

        response = get_llm("router").invoke(
            _SUMMARY_PROMPT.format(previous=prefix, transcript=transcript)
        )
        return str(response.content).strip()
    except Exception:
        # Summarisation failing shouldn't break the chat — keep what we had.
        return previous


# --------------------------------------------------------------------------
# Session store
# --------------------------------------------------------------------------

_SESSIONS: dict[str, ConversationMemory] = {}


def get_memory(session_id: str) -> ConversationMemory:
    """Fetch or create the memory for a session."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = ConversationMemory(session_id=session_id)
    return _SESSIONS[session_id]


def drop_memory(session_id: str) -> None:
    """Forget a session entirely."""
    _SESSIONS.pop(session_id, None)


# For durable sessions, swap `_SESSIONS` for LangGraph's `SqliteSaver`
# checkpointer, which persists the whole graph state per `thread_id`. Don't run
# both — two stores for "what was said" means two things to keep in sync.
