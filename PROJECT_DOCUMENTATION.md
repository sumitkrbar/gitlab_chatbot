# Project Documentation: GitLab Handbook Assistant

## 1. The problem

GitLab "builds in public": its Handbook and Direction pages are some of the most
detailed public records of how a company actually operates. But they're *huge* —
thousands of pages — so finding a specific answer ("what does GitLab mean by
*iteration*?", "how is engineering structured?") is slow. The goal was a chatbot
that lets employees and candidates *ask* the handbook in natural language and get
a trustworthy, sourced answer.

Trustworthiness is the main thing here. A chatbot that confidently invents GitLab
policy is worse than no chatbot, so the design is built around grounding,
transparency, and honest refusal rather than just producing fluent text.

> A note on the data: the brief points at `about.gitlab.com/direction/`, but
> GitLab has since retired that site (it now 301-redirects to `/whats-new/`). The
> product direction, strategy, and vision content has moved into the handbook
> under `/handbook/product`. I found this while looking into why "product
> direction" questions were retrieving the wrong pages, and re-pointed the crawler
> at the live sources (`/handbook/product` + `/whats-new`).

## 2. Approach — Retrieval-Augmented Generation (RAG)

Rather than fine-tuning or stuffing everything into a prompt, I used RAG:

1. **Ingest (offline):** crawl a curated set of GitLab pages → clean the text →
   split into heading-aware chunks → embed each chunk → store vectors in a local
   Chroma index.
2. **Answer (online):** embed the user's question → retrieve the top-k most
   similar chunks → pass *only those* to Gemini with strict instructions to answer
   from them and cite each claim.

This keeps answers current with the source pages, makes every claim auditable, and
lets the model say "I don't know" when the corpus doesn't cover something.

## 3. Key technical decisions (and why)

| Decision | Choice | Why |
|----------|--------|-----|
| **Frontend** | Streamlit | One Python codebase, native chat widgets, free one-click deploy. |
| **Generation LLM** | Google Gemini (`gemini-2.5-flash`, thinking off) | Strong free-tier model; reserved purely for answer generation. Thinking is disabled so the full token budget goes to the visible, cited answer. |
| **Embeddings** | **Local** `bge-small-en-v1.5` (fastembed/ONNX) | See below — chosen for quota-resilience. Uses bge's asymmetric query/passage representations for better recall. |
| **Vector store** | Local Chroma, **committed to the repo** | Zero external infra, free, and the deployed app boots instantly with no re-scrape — nothing to fail mid-demo. |
| **Data scope** | Curated subset via an allowlist | Low coverage for for now because of limited resources; widening is a one-line config change. |
| **Chunking** | Heading-aware, ~650 tokens, 100 overlap | Respects the handbook's structure so chunks are self-contained and citations map to real sections. |

**Why local embeddings.** I started with Gemini's hosted embeddings, but the
free-tier embedding quota turned out to be tiny and bulk ingestion kept hitting
`429 RESOURCE_EXHAUSTED`. The bigger problem is that the query path embeds on every
user turn, so tying retrieval to a rate-limited API would make the deployed app
fail intermittently for real users. I moved embeddings to a local ONNX model
(`bge-small-en-v1.5` via fastembed: no GPU, no API key, ~0.02s per query).
Retrieval is now free and unlimited, and the Gemini quota is only spent on
generating the answer, which is the part that actually needs it.

## 4. The pipeline

There are two entry points and a shared `src/` package. `ingest.py` builds the
index offline; `app.py` serves queries online. Both read their settings from
`src/config.py`, which is the single place models, paths, retrieval knobs, and the
crawl allowlist live.

### Ingestion (offline): `ingest.py`

`ingest.py` is a thin CLI wrapper that runs three stages and prints an honest
summary of each (pages kept, pages skipped as too short, errors).

1. **Crawl — `src/scraper.py:crawl()`.** Each URL in `SEED_URLS` gets an equal
   slice of the page budget and a BFS confined to its own subtree, so a link-heavy
   area (e.g. `/handbook/company`) can't starve the rest. Within a seed, each BFS
   level is fetched concurrently with a thread pool (`_fetch`), because handbook
   pages are slow to load. A link is only followed or kept if its URL passes the
   `ALLOWED_PREFIXES` check (`_is_allowed`), and URLs are normalised (`_normalise`)
   so the same page isn't crawled twice via fragments or trailing slashes.
   `_extract()` pulls clean main content as Markdown with `trafilatura` (so headings
   survive as `#`), and falls back to a stripped-down BeautifulSoup parse if that
   returns nothing. Pages under 200 characters are dropped as boilerplate. The
   output is a list of `Page(url, title, text)`.

2. **Chunk — `src/chunker.py:chunk_pages()`.** Each page is split on Markdown
   headings (`_split_into_sections`), then consecutive sections are packed into
   ~`CHUNK_TOKENS` (650) windows with `CHUNK_OVERLAP_TOKENS` (100) of overlap
   (`_pack`); a single section larger than the target is sliced into overlapping
   token windows. Token counts use `tiktoken`. Each `Chunk` carries a heading
   "breadcrumb" prepended to its text (this helps retrieval), source metadata
   (`url`, `title`, `heading`, `tokens`), and a content-hash id (`sha1` of
   url + heading + leading text). That id is what makes re-ingestion idempotent.

3. **Embed + index — `src/vectorstore.py:index_chunks()`.** Chunks are
   de-duplicated by id (within the batch and against what's already stored),
   embedded in batches with fastembed's `passage_embed`, and upserted into a
   persistent Chroma collection configured for cosine distance. Running with
   `--keep-index` upserts instead of rebuilding from scratch. The committed
   `data/chroma/` directory is the output of this step.

### Answering (online): `app.py` → `src/rag.py`

`app.py` is the Streamlit UI; the retrieval and generation logic lives in
`src/rag.py` so it stays testable and UI-agnostic. For each question:

1. **Input check — `guardrails.check_input()`.** Rejects empty or over-long input
   before anything is spent, and raises an injection flag if the text matches known
   jailbreak patterns. The flag doesn't block the answer — it's surfaced in the UI,
   and the system prompt is what actually resists the injection.

2. **Retrieve — `rag.retrieve()`.** If there's chat history and the question is
   short or pronoun-heavy (`_needs_rewrite`), `rewrite_followup()` asks Gemini to
   rewrite it into a standalone query ("how does *it* work?" becomes "how does
   GitLab's async communication work?"). The query is embedded with fastembed's
   `query_embed` and run against Chroma via `vectorstore.query()`, which converts
   cosine distance back into a 0–1 similarity. `guardrails.assess_retrieval()` then
   checks the best similarity: if it's below `MIN_SIMILARITY` the function returns a
   refusal and no LLM call is made. Everything is bundled into a `RagResult`.

3. **Generate — `rag.stream_answer()`.** When retrieval is strong enough, the
   retrieved chunks are formatted into a numbered `SOURCES` block
   (`prompts.build_user_turn`), combined with the recent chat history and the
   grounded `SYSTEM_PROMPT`, and streamed from Gemini token by token. Thinking is
   disabled for `2.5` models so the budget goes to the visible answer. Transient
   5xx errors are retried with exponential backoff, but only while nothing has been
   streamed yet, so the UI never sees duplicated text.

4. **Render — back in `app.py`.** The streamed text is accumulated and shown live
   with a cursor, `linkify_citations()` rewrites each `[n]` marker into a clickable
   link to that source's URL, and `render_sources()` lists the exact chunks (with
   similarity scores) in an expander. The turn is saved to `st.session_state` so
   history survives Streamlit's reruns.

### Data flow at a glance

```
ingest.py:  SEED_URLS -> crawl() -> [Page] -> chunk_pages() -> [Chunk] -> index_chunks() -> Chroma
app.py:     question -> check_input() -> retrieve() -- rewrite? -- query() -> assess_retrieval()
                                                                                 |
                                       refuse  <-- weak --        -- strong -->  stream_answer() -> Gemini
                                                                                 |
                                       linkify_citations() + render_sources()  <-+
```

## 5. What makes it trustworthy

**Transparency**
- **Inline citations:** every factual sentence carries a `[n]` marker, rendered as a
  clickable link to the exact source URL.
- **Sources panel:** a "📚 Sources" expander shows precisely which chunks the model
  was given, each with its similarity score — you can verify the answer against the
  source in one click.
- **Visible reasoning aids:** when a follow-up is rewritten, the UI shows the
  standalone question it actually searched ("🔁 Interpreted as: …").

**Guardrails**
- **Scope / grounding gate:** a deterministic retrieval-confidence check
  (`MIN_SIMILARITY`). If the best match is weak, the bot refuses politely and points
  to what it *can* answer — no LLM call, no hallucination. This catches both
  off-topic questions ("what's the weather?") and out-of-corpus ones.
- **Grounded system prompt:** forbids outside knowledge, mandates citations, and
  requires an explicit "I don't have that in the GitLab Handbook" when context is
  insufficient.
- **Prompt-injection resistance:** the system prompt refuses instructions embedded
  in sources or user input; suspicious inputs are flagged in the UI.
- **Input hygiene:** empty/over-long inputs are rejected before any API spend.

**Product thinking**
- Streaming answers, suggested starter questions, live index stats in the sidebar,
  clear-chat, and friendly handling of rate limits / API errors so the experience
  degrades gracefully instead of crashing. Transient 5xx/overloaded errors (common
  on the free tier) are retried automatically with exponential backoff before any
  error is shown.

## 6. Measuring it

`eval/run_eval.py` runs a gold set (`eval/questions.yaml`) and reports:
- **Retrieval hit-rate@k** — for on-topic questions, does an expected source page
  appear in the top-k? (Did we retrieve the right thing?)
- **Guardrail precision** — for deliberately off-topic questions, does the scope
  gate correctly refuse?

This turns "it feels good" into numbers and makes regressions visible when the
allowlist, chunking, or thresholds change.

**Results on the current index** (~520 chunks across 7 GitLab sections):
- **Retrieval hit-rate@5 = 11/11 (100%)** — every on-topic question retrieves a
  correct source page (on-topic similarities cluster at 0.78–0.86).
- **Guardrail refusal = 3/3 (100%)** on off-topic questions.

The eval also drove a concrete tuning decision: off-topic queries top out around
0.64 similarity while on-topic ones start at 0.78, so `MIN_SIMILARITY` is set to
0.70 — the clean gap between them. Notably, *"write me a Python script"* scores 0.64
(GitLab's handbook is full of engineering content), and the threshold correctly
refuses it rather than answering an off-topic code request.

## 7. Trade-offs & limitations

- **Curated, not exhaustive:** tuned for answer precision; some handbook areas
  aren't indexed yet (widen the allowlist to add them).
- **Dense-only retrieval:** a hybrid BM25 + dense re-ranker would help on rare exact
  keywords (e.g. an internal acronym).
- **Static snapshot:** the handbook changes often; a scheduled re-ingest would keep
  the index fresh.
- **Confidence threshold is a blunt instrument:** a learned router or an
  LLM-as-judge groundedness check would be more nuanced than a single cosine cutoff.
- **Generation depends on the Gemini free tier:** retrieval is fully local and
  unlimited, but answer generation uses Gemini's free quota (1,500 requests/day).
  Heavy testing can exhaust it for the day; the app degrades gracefully with a
  clear rate-limit message, and a fresh key or the next-day reset restores it.

## 8. What I'd do next

1. Hybrid retrieval + a cross-encoder re-ranker for higher precision.
2. LLM-as-judge groundedness scoring in the eval loop (catch unsupported claims, not
   just bad retrieval).
3. Scheduled re-ingest (GitHub Action) to track handbook changes.
4. Per-answer feedback (👍/👎) logged to improve the gold set over time.
