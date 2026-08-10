# Oracle — Engineering Decisions and Trade-offs

A record of the choices made while building Oracle, including the alternatives
that were considered and rejected.

## Why a graph instead of a single RAG call

A single retrieve-then-generate call is simpler and would answer most questions
about a resume adequately. The graph earns its complexity in two places:
questions that need external information the documents cannot contain, and
questions the corpus genuinely does not cover, where the correct behaviour is
to say so rather than stretch weak evidence into an answer.

The secondary benefit is inspectability. Because each node's contribution is a
discrete state update, the interface can display the route taken, the nodes
executed, the tools invoked, and the chunks retrieved with their scores. The
retrieval behaviour is visible rather than a black box.

## Why the router falls back to retrieval rather than direct answering

An unreachable router, unparseable JSON, or a hallucinated tool name all route
to document retrieval. The alternative — falling back to answering directly —
would mean the failure mode of the routing layer is an ungrounded answer.
Grounded search that returns nothing is a recoverable failure; a confident
fabrication is not.

## Why the similarity threshold discards results

Vector search always returns something. Without a floor, an off-topic question
retrieves the five least-irrelevant chunks in the corpus, and a language model
handed five irrelevant chunks will generally find a way to use them.

Discarding below-threshold results converts that silent failure into an
explicit one. It also creates the signal the retry loop and the "not in my
documents" response both depend on. The trade-off is that a threshold set too
high produces false negatives on legitimate questions, which is why it is
exposed as a slider in the interface rather than buried in configuration.

## Why chunking is token-based and paragraph-aware

Character-based chunking is simpler but the chunk sizes do not correspond to
what the embedding model processes, so the effective size drifts with the
content. Token-based sizing keeps them aligned.

Packing whole paragraphs rather than cutting at a fixed offset matters
specifically for resume content, where a bullet point split across two chunks
loses the association between an achievement and its context. The cost is
slightly uneven chunk sizes.

## Why the calculator validates an AST instead of using a parser library

Calling eval on a string produced by a language model is arbitrary code
execution. The model is handed a channel through which any input it can be
induced to emit becomes running code, and in a publicly deployed application
that input originates from anonymous visitors.

The first implementation avoided eval by parsing with SymPy, on the assumption
that a symbolic maths library builds an expression tree rather than executing
code. That assumption was wrong, and the test suite caught it. SymPy's
expression parser calls eval internally on the tokenised input, so a payload
of the form `__import__('os').system(...)` executed and its side effect landed
before the result failed to convert into a symbolic expression. The tool
reported failure while the command had already run — the most dangerous shape
a security bug can take, because the error message looks like the defence
working.

The lesson generalised beyond the fix: a test asserting that hostile input
*returns an error* proves nothing about whether it *executed*. The check was
rewritten to have payloads attempt to create a sentinel file and to fail if
that file appears, which tests the property that actually matters.

The replacement parses the expression with Python's own ast module and walks
the tree, permitting only numeric literals, a fixed set of arithmetic
operators, and an explicit whitelist of mathematical functions. Attribute
access, subscripts, string literals, lambdas, and comprehensions are rejected
during validation, so no path exists to a callable that was not deliberately
allowed. Exponents and factorials are capped, since both are trivial to write
and expensive enough to evaluate that an unbounded one is a denial of service.

## Why the entire corpus is not simply placed in the context window

Llama 3.1 supports a context window large enough to hold a resume, a set of
project writeups, and a biography several times over. For a corpus this size,
stuffing everything into the prompt is a legitimate approach and would produce
competitive answers.

Retrieval was chosen because the corpus is intended to grow, because per-query
cost and latency scale with prompt size, and because citation to specific
source documents is a requirement rather than a nice-to-have. The stuffed-context
approach remains the honest benchmark: if retrieval does not beat it, the
retrieval configuration is wrong rather than the model.

## Why the provider is abstracted rather than committing to one

Local inference and public deployment are genuinely incompatible requirements
on free hosting. Committing to local execution means no live demo. Committing
to a hosted API means giving up the privacy property and the no-API-key
property that motivated local execution in the first place.

Abstracting the provider preserves both. The same weights answer in both
environments, so the behaviour observed during development is the behaviour
deployed, and neither claim about the system needs qualification.

## Why conversation memory rewrites queries rather than concatenating history

Prepending raw conversation history to the retrieval query dilutes the
embedding — the vector drifts toward an average of everything discussed rather
than what is being asked now. Rewriting produces a single focused query while
still resolving the references that made the follow-up unretrievable.

The rewrite is guarded: it returns the original question unchanged on any
failure, and rejects rewrites that come back disproportionately long, which is
the characteristic failure mode of a small model that decided to answer the
question instead of rewriting it.

## Known limitations

Retrieval quality tracks the corpus more than the model. A short document set
produces thin answers regardless of which language model generates them, and
prose explaining reasoning embeds substantially better than bullet fragments.

The highest-value improvements not yet implemented are a cross-encoder reranker
over a wider candidate set, and hybrid keyword-plus-dense search. Dense
embeddings blur exact technology names together, which matters when the
questions being asked are literally about specific named tools.

Conversation memory is held in process and does not survive a restart. Making
it durable would mean adopting the graph checkpointer, which persists complete
graph state per session, rather than maintaining a second store for the message
list alongside it.
