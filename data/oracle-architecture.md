# Oracle — Architecture Deep Dive

## The graph

Oracle orchestrates four agents as a LangGraph state machine. State is a single
TypedDict; each node returns a partial update that LangGraph merges, which is
what allows the router's decision, the retrieved chunks, and the tool results
to accumulate independently before synthesis reads all of them.

The entry point is the router. Two conditional edges control flow: one out of
the router selecting the destination, and one out of synthesis deciding whether
to stop or retry retrieval.

## Router agent

The router makes a single constrained call and returns JSON naming one of three
destinations: `research` for anything about the person's experience, skills,
projects, or history; `tool` for live information, arithmetic, or external
facts; and `direct` for greetings and questions about the chatbot itself.

Small local models wrap JSON in prose or code fences even when instructed not
to, so the response is parsed defensively — first looking for a fenced block,
then for any brace-delimited object, and falling back to document retrieval if
neither yields valid JSON. Retrieval is also the fallback when the model is
unreachable or names a tool that does not exist. The reasoning is that grounded
search failing visibly is always better than an ungrounded answer that sounds
confident.

Router temperature is pinned to zero. The task is picking one label from a
fixed set, and sampling only introduces mistakes.

## Research agent

The research node embeds the query with the same sentence-transformers model
used at ingestion and queries ChromaDB for nearest neighbours. The collection
is created with cosine distance, and embeddings are L2-normalised at write
time, so similarity is recovered as one minus the returned distance.

Candidates below the similarity threshold are discarded rather than returned as
the least irrelevant option. This makes an empty result meaningful: it is the
signal the graph uses to either retry with a wider net or tell the user plainly
that the documents do not cover the question.

On a retry pass the node doubles top-k and relaxes the threshold rather than
re-running an identical query that already failed.

## Tool agent

Three tools are available. Web search runs through DuckDuckGo and needs no API
key. Wikipedia lookup returns a short summary plus the article URL, and
surfaces disambiguation options when a topic is ambiguous. The calculator
parses expressions into a Python AST and evaluates only nodes it has explicitly
whitelisted, so untrusted input cannot reach the interpreter.

Every tool returns a uniform result object and never raises. Network failures,
rate limits, and empty result sets all come back as a failed result carrying a
readable error, which the synthesis agent sees as an observation. A
rate-limited search degrades the answer rather than crashing the request.

Tool schemas are derived programmatically from each function's signature and
docstring, so the description the model sees cannot drift away from what the
function actually accepts.

## Synthesis agent

Synthesis composes the final answer from retrieved evidence, tool output, and
conversation context. The prompt constrains it to the supplied evidence and
forbids inventing employers, dates, technologies, or numbers. It is instructed
that stating plainly that something is not in the documents is a correct
answer, not a failure.

Citations are bracketed numbers matching the numbered evidence block. After
generation, the answer is scanned for citation markers and only the sources the
model actually referenced are surfaced in the interface, so an answer citing
two of five retrieved chunks does not display five sources.

## Retry loop

The conditional edge out of synthesis returns to research only when the route
was document retrieval, no chunks were found, and the iteration count is below
the configured cap. All three conditions must hold, so the graph cannot spin.

## Ingestion

Documents are discovered recursively from the data directory across PDF, DOCX,
Markdown, and plain text. Each format is reduced to text segments — one per
page for PDFs, one per top-level heading for Markdown — so the chunker never
needs to know about file formats.

Markdown segments retain their heading as a prefix. A chunk reading "built with
Kafka and Redis" is substantially more retrievable when it still carries the
project title above it.

Chunking is token-based using tiktoken rather than character-based, so chunk
sizes correspond to what the embedding model actually processes. Whole
paragraphs are packed until the token budget is reached rather than cutting at
a fixed offset, which keeps resume bullets and code blocks intact. When a chunk
boundary is crossed, the tail of the previous chunk is carried forward as
overlap so a fact spanning the boundary remains retrievable from both sides. A
single paragraph exceeding the budget is hard-split on token boundaries as a
fallback.

Chunk identifiers are a truncated SHA-1 hash of the source path and the chunk
index within that file, which makes re-ingestion idempotent — running the
pipeline twice updates existing vectors instead of duplicating the corpus.

## Conversation memory

The last several turns are kept verbatim. Once that window overflows, evicted
turns are compressed into a running summary by a single cheap model call, and
the summary is prepended to subsequent prompts. The full transcript is retained
separately for display; only the prompt window rolls.

Before retrieval, follow-up questions are rewritten into standalone form.
Retrieval is stateless, so "what about that one?" embeds to nothing useful. A
small model resolves pronouns and references against the conversation history,
and the original question is returned unchanged on any failure or if the
rewrite comes back implausibly long — a bad rewrite is worse than no rewrite.

Sessions are keyed per browser tab, so two simultaneous visitors never share
history.

## Deployment

The provider is abstracted behind a single module. Local development runs
Ollama with Llama 3.1 on the developer's own machine, with no API keys and no
data leaving the box. Streamlit Community Cloud provides roughly a gigabyte of
RAM and no GPU, which cannot host an 8-billion-parameter model, so the deployed
instance points at Groq — which serves the same Llama 3.1 weights over an API.

The consequence is that the deployed demo runs the identical model as local
development, and switching between them is one environment variable. No other
module in the codebase knows where inference happens.

The embedding model was chosen partly for this constraint: at 384 dimensions
and roughly eighty megabytes, it runs comfortably on free-tier CPU hosting,
where a larger embedding model would not.

Because the deployed filesystem is ephemeral, the application rebuilds its
vector index at startup when it finds the store empty. For a corpus of this
size that takes seconds, and it makes the deployment self-healing across
restarts.
