"""Provider adapter — the only module that knows where the LLM actually runs.

Oracle is local-first: `ollama` serving `llama3.1` on your own machine, no API
keys, no data leaving the box. But Streamlit Community Cloud has ~1GB of RAM
and no GPU, so the *deployed* demo cannot run Ollama.

Groq serves open-weight models over an API. Which one is decided at runtime by
`resolve_groq_model()` rather than hardcoded, because hosted ids are retired —
`llama-3.1-8b-instant` was deprecated for free tiers in June 2026 and the
deployed app started 404ing on every question. Llama is preferred whenever the
account can serve it, so the local and hosted models match where possible.
Every other module asks for `get_llm()` and stays provider-agnostic.

    ORACLE_PROVIDER=ollama   # local dev (default)
    ORACLE_PROVIDER=groq     # deployed; needs GROQ_API_KEY
"""

from __future__ import annotations

import functools

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings


@functools.lru_cache(maxsize=8)
def get_llm(role: str = "synthesis", json_mode: bool = False) -> BaseChatModel:
    """Return a chat model for `role` ("router" | "synthesis" | "tool").

    Cached per (role, json_mode) — constructing a client per call is wasteful,
    and for Ollama it re-negotiates the connection every time.
    """
    model = settings.router_model_name if role == "router" else settings.model_name
    # The router picks one word from a fixed set; sampling only adds mistakes.
    temperature = 0.0 if role == "router" else settings.temperature

    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_host,
            temperature=temperature,
            num_ctx=settings.num_ctx,
            format="json" if json_mode else None,
        )

    if not settings.groq_api_key:
        raise RuntimeError(
            "ORACLE_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys, then put it in "
            ".env (local) or Streamlit secrets (deployed)."
        )

    from langchain_groq import ChatGroq

    # Resolved, not configured: the sidebar and the actual request must name the
    # same model, or a green status line sits above a 404.
    resolved, _ = resolve_groq_model()

    return ChatGroq(
        model=resolved or settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        timeout=settings.request_timeout,
        model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
    )


def health_check() -> tuple[bool, str]:
    """Verify the active provider is reachable and the model is available.

    Returns `(ok, message)`. The UI calls this on startup so a stopped Ollama
    shows a clear banner instead of a connection traceback on first question.
    """
    import httpx

    if settings.provider == "groq":
        if not settings.groq_api_key:
            return False, (
                "GROQ_API_KEY is not set. Add it to Streamlit secrets "
                "(top level, not under a section) and reboot the app."
            )

        model, note = resolve_groq_model()
        if model is None:
            return False, note
        return True, f"Groq · {model}" + (f" — {note}" if note else "")

    try:
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception:
        return False, (
            f"Cannot reach Ollama at {settings.ollama_host}. "
            "Start it with `ollama serve` (or launch the Ollama desktop app)."
        )

    tags = [m.get("name", "") for m in resp.json().get("models", [])]
    wanted = settings.ollama_model
    # `ollama list` reports "llama3.1:latest" for a plain "llama3.1" pull.
    if not any(t == wanted or t.split(":")[0] == wanted.split(":")[0] for t in tags):
        return False, f"Model '{wanted}' not pulled. Run: ollama pull {wanted}"

    return True, f"Ollama · {wanted}"


# --------------------------------------------------------------------------
# Groq model discovery
# --------------------------------------------------------------------------
#
# Hardcoding a hosted model id is a slow-motion outage. `llama-3.1-8b-instant`
# is listed as a current production model in Groq's own docs, yet this account
# gets a 404 for it — access differs per account, and ids are retired over
# time. So the model is discovered at runtime: use the configured one when the
# account can serve it, otherwise fall back through a preference order and say
# plainly in the UI that a substitution happened.

# Llama first, so the "same model locally and deployed" property holds whenever
# the account can serve it at all.
_GROQ_PREFERENCE = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "groq/compound-mini",
    "groq/compound",
]

# Substrings marking models that cannot answer a chat prompt: speech, embedding
# and the prompt-guard safety classifiers.
_NOT_CHAT = ("whisper", "tts", "embed", "guard", "rerank", "moderation")


def _is_chat_model(model_id: str) -> bool:
    return not any(marker in model_id.lower() for marker in _NOT_CHAT)


@functools.lru_cache(maxsize=1)
def available_groq_models() -> tuple[str, ...]:
    """Model ids this Groq account can actually serve. Empty on failure."""
    import httpx

    try:
        resp = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            timeout=8.0,
        )
        resp.raise_for_status()
    except Exception:
        return ()
    return tuple(sorted(m.get("id", "") for m in resp.json().get("data", []) if m.get("id")))


def resolve_groq_model() -> tuple[str | None, str]:
    """Pick a usable Groq chat model.

    Returns `(model_id, note)`. `model_id` is None when nothing usable exists,
    and `note` then explains why. When a substitute is chosen the note names
    what was asked for, so the swap is never silent.
    """
    models = available_groq_models()
    if not models:
        return None, "Cannot reach Groq, or the API key was rejected."

    if settings.groq_model in models:
        return settings.groq_model, ""

    chat = [m for m in models if _is_chat_model(m)]
    if not chat:
        return None, (
            f"This Groq account serves no chat models "
            f"(available: {', '.join(models[:5])})."
        )

    for candidate in _GROQ_PREFERENCE:
        if candidate in chat:
            return candidate, f"'{settings.groq_model}' unavailable on this account"

    return chat[0], f"'{settings.groq_model}' unavailable on this account"


def active_model_name() -> str:
    """Model actually used for generation, for display in the UI."""
    if settings.provider != "groq":
        return settings.model_name
    resolved, _ = resolve_groq_model()
    return resolved or settings.groq_model
