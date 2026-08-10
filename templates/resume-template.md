# Resume / portfolio template

This file lives **outside `data/` on purpose** — placeholder text must never be
indexed, or the bot will retrieve and cite `[Your Name]` as if it were fact.

Fill in a copy, save it as `data/resume.md`, then run:

```
python -m src.ingest --rebuild
```

Everything below is a prompt for you to answer, not content to keep.

---

## What makes a good corpus here

Retrieval quality tracks what you write far more than which model generates the
answer. Two rules govern everything below.

**Write prose, not bullet fragments.** `• Kafka, Redis, PostgreSQL` embeds
almost identically to every other tech-stack list on earth, so it retrieves for
nothing in particular. "I moved the ingestion path from RabbitMQ to Kafka
because we needed replay after the consumer crashed" embeds distinctly and
retrieves for questions about message queues, reliability, and debugging.

**Answer the question behind the question.** Recruiters ask "has this person
handled scale?", not "list technologies." Write the paragraph that answers the
real question, and retrieval will find it.

---

## data/resume.md

```markdown
# [Your Name]

[One paragraph: what you do, how long you've done it, what you're looking for.
Write it as speech, not as a headline.]

## Experience

### [Job title] — [Company], [Month Year] – [Month Year or Present]

[Two or three sentences on what the company does and what your team owned.]

[A paragraph per significant piece of work. What was the problem, what did you
build, what was the outcome? Include numbers where you have them — latency
before and after, users served, cost saved. Numbers are what makes an answer
specific enough to be worth reading.]

[Repeat per role.]

## Education

### [Degree] — [Institution], [Year]

[Relevant coursework, thesis, or projects — one or two sentences. Skip if it's
been a while and your work speaks louder.]

## Skills

[Group them and say how you've used each, rather than listing. "Python — six
years, primarily backend services and data pipelines" answers more questions
than "Python" does.]
```

---

## data/projects.md

One section per project. This is usually the highest-value file in the corpus,
because project questions are what people actually ask.

```markdown
# Projects

## [Project name]

**Problem.** [What was broken or missing? Why did this need to exist?]

**What I built.** [The system, in a paragraph. Concrete enough that someone
could picture the architecture.]

**Technical decisions.** [The interesting part. What did you choose, what did
you reject, and why? This is what distinguishes you from everyone else who
listed the same framework.]

**Outcome.** [What happened. Shipped to how many people, what improved, what
you learned. Including what didn't work reads as honest, not weak.]

**Stack.** [Technologies, at the end, once the reader already cares.]
```

---

## data/bio.md

```markdown
# About [Your Name]

## Short bio
[Two sentences. The version that goes in a conference programme.]

## Long bio
[Three paragraphs: how you got into this work, what you've focused on, where
you're heading.]

## What I'm looking for
[Role type, domains that interest you, team size and working style that suits
you. Be specific — this is a question recruiters ask constantly and a static
resume never answers.]
```

---

## data/faq.md

The questions you're tired of retyping. Consider this file mandatory — it's the
one that saves you the most time.

```markdown
# FAQ

## Are you open to relocation?
[Answer.]

## What's your notice period?
[Answer.]

## What are you looking for in your next role?
[Answer.]

## Why are you leaving your current role?
[Answer — the version you're happy having repeated verbatim to strangers.]

## What's your experience with [the technology you're always asked about]?
[Answer.]
```

---

## Before you deploy

Everything in `data/` is committed to the repository and served publicly by the
deployed app. Anyone can ask for it, and the bot will read it back.

Leave out home addresses, phone numbers, reference contacts, salary figures,
government identifiers, and anything about a former employer you would not say
on a public stage. A published portfolio bot is a public document with a chat
interface, not a private one.
