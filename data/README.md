# data/

Drop the documents you want Oracle to answer from in this folder, then run:

```
python -m src.ingest
```

## Supported formats

`.pdf`, `.docx`, `.md`, `.txt`

## What to put here

| File | Why it helps |
|---|---|
| `resume.pdf` | The core document — experience, skills, education |
| `projects.md` | One section per project: problem, stack, your role, outcome |
| `bio.md` | Short and long bio, elevator pitch, what you're looking for |
| `faq.md` | Q&A you're tired of retyping: visa status, notice period, relocation |
| `case-studies/` | Deeper writeups — these give retrieval something substantial to chunk |

Retrieval quality tracks the corpus more than the model. A one-page resume
gives roughly ten chunks, and the answers stay thin no matter how good the
prompt is. Prose with context ("I chose Kafka over RabbitMQ because…") embeds
far better than bullet fragments.

## Privacy

`.gitignore` excludes everything in this folder except this README. If you want
your resume tracked in git, remove the `data/*` lines from `.gitignore`.
Keep phone numbers and home addresses out of the corpus regardless — anything
ingested here can be surfaced verbatim by the bot to anyone who asks.
