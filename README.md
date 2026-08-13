# Oracle

A multi-agent RAG system that answers questions about my resume, projects, and
portfolio. A router agent decides whether to search my documents, call a tool,
or answer directly; a synthesis agent writes the answer with citations back to
the source document — and the UI shows the whole path it took.

Runs local-first on Ollama with **Llama 3.1**, no API keys and no data leaving
the machine. The public demo runs the same model on Groq, because Streamlit
Community Cloud has no GPU — one environment variable switches between them.

**[Live demo →](#)** · _(add your URL after deploying)_

---

## Architecture

```
                        ┌── retry once, wider net ──┐
                        ↓                           │
                  ┌──────────┐   "research"   ┌──────────┐
   Streamlit UI ──│  ROUTER  │───────────────→│ RESEARCH │──┘
     (app.py)     └────┬─────┘                └────┬─────┘
                       │                           │ found evidence
              "tool"   │                  ChromaDB │
                       ↓                  (MiniLM) ↓
                 ┌──────────┐              ┌─────────────┐
                 │   TOOL   │─────────────→│  SYNTHESIS  │──→ answer
                 └──────────┘              └─────────────┘      + citations
              web search /                        ▲
              calculator /              "direct"  │
              wikipedia   ──────────────────────--┘
```

The retry decision sits on the edge out of **research**, not synthesis.
Synthesis is by far the most expensive node (~17s on a local 8B model), so
deciding afterwards meant paying for a full generation and discarding it.

| Layer | Choice | Why |
|---|---|---|
| LLM | Llama 3.1 — Ollama locally, Groq deployed | Same weights either way; adapter in `src/llm.py` |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, ~80MB, fast on CPU — runs on free-tier hosting |
| Vector DB | ChromaDB (persistent, cosine) | Zero-config, embedded, no server to run |
| Orchestration | LangGraph | Explicit state machine, so the trace is inspectable |
| UI | Streamlit | Chat plus live agent-trace visualisation |

**Agents**

- **Router** — one constrained JSON call classifying the question as
  `research` / `tool` / `direct`. Falls back to `research` on any parse or
  connection failure, since RAG is the safe default here.
- **Research** — embeds the query and pulls the top-k chunks from Chroma.
  Below-threshold hits are dropped, so "no evidence" is a real signal rather
  than five irrelevant chunks.
- **Tool** — dispatches web search, Wikipedia, the calculator, or the calendar
  tool (which returns a downloadable `.ics`, not an API write — the app is
  public, so a tool with calendar credentials would let any visitor create
  events in a real account). Tool failures become observations, never
  exceptions. Results the user will act on, like a scheduled time, are returned
  verbatim rather than paraphrased through the model.
- **Synthesis** — writes the answer from evidence only, with `[n]` citations
  the UI expands back into sources. Only citations the answer actually
  referenced are surfaced.

Conversation memory rewrites follow-ups into standalone questions, since
retrieval is stateless and "what about that one?" embeds to nothing useful.
The rewrite is guarded on both sides: skipped entirely when the question
already retrieves well, and discarded if it retrieves worse than what the user
typed. Both checks are embedding lookups — milliseconds against a multi-second
LLM call.

---

## Setup

### 0. Prerequisites

- **Python 3.11+**
- **~6 GB free disk** — llama3.1:8b is ~4.7 GB plus dependencies
- **8 GB RAM minimum**, 16 GB comfortable

### 1. Install Ollama

**Windows / macOS** — download from [ollama.com/download](https://ollama.com/download)
and run the installer. On Windows it installs as a background service that
starts at login.

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the model

```bash
ollama pull llama3.1
```

Under 16 GB of RAM, a quantised build is much lighter — set
`ORACLE_OLLAMA_MODEL` to match if you use a different tag:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 3. Confirm Ollama is running

**Oracle assumes a server is already listening on `http://localhost:11434`.**
Nothing in this project starts it for you.

```bash
ollama list
```

If that prints your models, you're set. If not, start it:

```bash
ollama serve
```

On Windows the desktop app runs the server in the background — look for the
icon in the system tray. The app shows a red banner in the sidebar when it
can't reach Ollama, so you'll know immediately rather than on first question.

### 4. Install dependencies

**Windows (PowerShell)**

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

**macOS / Linux**

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

First install pulls PyTorch — a few hundred MB and a few minutes. The CPU build
is fine.

### 5. Add your documents

`data/` ships with three documents describing this project's own architecture
and engineering decisions, so the system has a real corpus and runs end to end
out of the box. Add your resume, project writeups, bio, and FAQ alongside them.

[templates/resume-template.md](templates/resume-template.md) has scaffolds for
`resume.md`, `projects.md`, `bio.md`, and `faq.md`, plus notes on what makes
text retrieve well. It sits outside `data/` deliberately: placeholder text must
never be indexed, or the bot will cite `[Your Name]` as fact.

**Placeholder guard.** Any `.md`/`.txt` file in `data/` containing the marker
`SAMPLE-DATA:` in its first few lines is detected by
`src.ingest.sample_documents()`, and the app then shows a permanent banner
telling every visitor the content is not real. Put that marker in draft files
while you're writing them — a portfolio bot confidently reciting placeholder
biography is the worst failure this project can have. Remove the marker when
the content is genuinely yours and the banner disappears.

### 6. Ingest

```bash
python -m src.ingest
```

First run also downloads the MiniLM weights (~80 MB) and writes the vector
store to `chroma/`. The app auto-ingests at startup if the index is empty, so
this step is optional — but running it manually shows you the chunk counts.

Use `--rebuild` after replacing documents, which wipes the collection first:

```bash
python -m src.ingest --rebuild
```

### 7. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

You can also query the graph straight from the terminal, which is the fastest
way to debug retrieval:

```bash
python -m src.agents "What did they build with LangGraph?"
```

### Verifying the install

`smoke_test.py` exercises the whole pipeline — chunking and overlap, embedding,
the Chroma round-trip, retrieval scoring, threshold rejection of off-topic
queries, all three tools, memory, and graph compilation. It reports LLM
connectivity separately, so it's useful before Ollama is even installed:

```bash
python smoke_test.py
```

It includes a security check that feeds the calculator hostile payloads and
fails if any of them executes. That test exists because it caught a real
vulnerability — see the comment above `calculator` in `src/tools.py`.

---

## Deploying the public demo

Ollama can't run on Streamlit Community Cloud (no GPU, ~1 GB RAM), so the
deployed instance points at Groq — which serves the same Llama 3.1 weights,
free, and fast.

1. **Get a free Groq key** at [console.groq.com/keys](https://console.groq.com/keys).
2. **Replace the demo documents** and commit your `data/` folder. The app
   rebuilds its index at boot and the deployed filesystem is ephemeral.
   Anything in there is public — no phone numbers, addresses, or reference
   contacts.
3. **Push to GitHub**, then create an app at
   [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`.
4. **Add secrets** in the app's settings (⋮ → Settings → Secrets), then reboot
   the app:

   ```toml
   ORACLE_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_..."
   ```

   Keys must be **top level**, not nested under a `[section]` — `app.py` copies
   top-level secrets into the environment before `src.config` is imported, and
   nested tables are skipped.

That's the only difference between the two environments. Locally you stay on
Ollama; `src/llm.py` is the only module that knows which is which.

**If the deployed app says "Cannot reach Ollama at localhost":** the secrets
aren't reaching the config. Either `ORACLE_PROVIDER` isn't set, it's nested
under a section, or the app wasn't rebooted after saving. The sidebar shows
which provider is actually live, so it will read `Groq · llama-3.1-8b-instant`
once it's working.

Two deployment details this repo already handles, both of which fail in
confusing ways otherwise:

- **Secrets are not environment variables.** `src/config.py` builds its
  settings at import time from the environment, and Streamlit Cloud exposes
  secrets only through `st.secrets`. `app.py` bridges them across *before*
  importing config — otherwise the deployed app silently keeps the local Ollama
  defaults.
- **SQLite is too old.** Chroma needs SQLite >= 3.35 and the Streamlit Cloud
  image ships an older one. `requirements.txt` pulls `pysqlite3-binary` on Linux
  only, and `app.py` swaps it in at startup.

---

## Configuration

Everything lives in [src/config.py](src/config.py) and can be overridden with
`ORACLE_`-prefixed environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `ORACLE_PROVIDER` | `ollama` | `ollama` or `groq` |
| `ORACLE_OLLAMA_MODEL` | `llama3.1` | Must match your `ollama pull` tag |
| `ORACLE_GROQ_MODEL` | `llama-3.1-8b-instant` | Same family, hosted |
| `ORACLE_ROUTER_MODEL` | *(main model)* | Set a smaller model to speed up routing |
| `ORACLE_EMBED_MODEL` | `all-MiniLM-L6-v2` | Changing this requires deleting `chroma/` |
| `ORACLE_TOP_K` | `5` | Chunks per query |
| `ORACLE_SCORE_THRESHOLD` | `0.25` | Cosine floor; higher = stricter |
| `ORACLE_CHUNK_SIZE` / `_OVERLAP` | `800` / `120` | Tokens |

---

## Project structure

```
oracle/
├── app.py                 # Streamlit chat UI + agent-trace panel
├── smoke_test.py          # end-to-end pipeline check, LLM optional
├── data/                  # your documents (committed, for deployment)
├── templates/             # resume/project scaffolds — NOT indexed
├── src/
│   ├── config.py          # all tunables
│   ├── llm.py             # provider adapter: Ollama <-> Groq
│   ├── ingest.py          # discover -> load -> chunk -> embed -> Chroma
│   ├── retrieve.py        # query -> scored, filtered chunks
│   ├── agents.py          # LangGraph: router, research, tool, synthesis
│   ├── tools.py           # web search, Wikipedia, sandboxed calculator
│   └── memory.py          # rolling window, summarisation, follow-up rewriting
├── .streamlit/config.toml # watcher + theme
├── chroma/                # persisted vectors (gitignored, rebuildable)
└── requirements.txt
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Sidebar shows "Cannot reach Ollama" | Server not running | `ollama serve`, or start the desktop app |
| `Model 'llama3.1' not pulled` | Model missing | `ollama pull llama3.1` |
| Very slow first question | Model cold-loading into RAM | Normal; later turns are faster |
| "Couldn't find anything in the documents" for everything | Empty index | `python -m src.ingest --rebuild` |
| Chroma dimension mismatch | Embedding model changed after ingest | Delete `chroma/` and re-ingest |
| Answers ignore obvious facts | Threshold too strict | Lower the similarity slider in the sidebar |
| `ImportError` after editing `src/` | Streamlit cached the old module | Restart the server |
| `ollama` not recognised (Windows) | Not on PATH | Reopen the terminal after installing |
| `CUDA error: shared object initialization failed` | Ollama's CUDA runner crashing on this GPU | Use the Vulkan runner — see below |

### CUDA crash on NVIDIA laptop GPUs

On some hybrid-graphics machines (an NVIDIA discrete GPU alongside integrated
AMD/Intel graphics), Ollama's CUDA runners crash on model load:

```
CUDA error: shared object initialization failed
  in function ggml_cuda_kernel_can_use_pdl
llama-server terminated  exit status 0xc0000409
```

Both the `cuda_v12` and `cuda_v13` runners fail identically, so it isn't a
runner-version problem. Ollama also ships a Vulkan backend that drives the same
GPU without CUDA, and it works:

```bash
setx OLLAMA_LLM_LIBRARY vulkan
```

Restart Ollama afterwards. `OLLAMA_LLM_LIBRARY=cpu` is the fallback if Vulkan
also fails — correct but several times slower. Unset the variable to go back to
automatic selection.

---

## Notes on retrieval quality

Answer quality tracks the corpus more than the model. A one-page resume yields
about ten chunks and produces thin answers no matter which LLM is behind it.
Prose with context ("I chose Kafka over RabbitMQ because…") embeds far better
than bullet fragments.

The highest-value upgrades, in order: more written material in `data/`, a
cross-encoder reranker over the top-20, and hybrid BM25 + dense search —
keyword matching matters a lot for exact tech names that embeddings blur
together.
