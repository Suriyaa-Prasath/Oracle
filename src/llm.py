"""Provider adapter — the only module that knows where the LLM actually runs.

Oracle is local-first: `ollama` serving `llama3.1` on your own machine, no API
keys, no data leaving the box. But Streamlit Community Cloud has ~1GB of RAM
and no GPU, so the *deployed* demo cannot run Ollama.

Groq serves the same Llama 3.1 weights over an API, so switching providers
changes where inference happens without changing which model answers. Every
other module asks for `get_llm()` and stays provider-agnostic.

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

    return ChatGroq(
        model=settings.groq_model,
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
    if settings.provider == "groq":
        if not settings.groq_api_key:
            return False, "GROQ_API_KEY is not set."
        return True, f"Groq · {settings.groq_model}"

    import httpx

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
