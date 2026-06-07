# GitLab Handbook Assistant

A chatbot that answers questions about how GitLab works, grounded in GitLab's
public [Handbook](https://handbook.gitlab.com/) and
[Direction](https://about.gitlab.com/direction/) pages.

It's a Retrieval-Augmented Generation (RAG) app. Your question is embedded with a
local embedding model, matched against a vector index of curated GitLab pages, and
answered by Google Gemini using only the retrieved text. Answers come with inline
citations and a "Sources" panel, and the bot is built to say *"I don't have that in
the Handbook"* rather than make something up.

One design note up front: retrieval embeddings run locally
(`bge-small-en-v1.5` via fastembed/ONNX, no GPU, no API key). Every query has to be
embedded, so keeping that path local makes it free and avoids hitting Gemini's tiny
free-tier embedding quota on every turn. The Gemini API is used only for generating
the answer.

## What it does

- **Grounded answers.** Every factual sentence carries a `[n]` citation linked to
  its source URL.
- **Transparency.** A "Sources" expander under each answer shows the exact chunks
  retrieved, with similarity scores.
- **Guardrails.** Off-topic or out-of-corpus questions are refused via a
  retrieval-confidence gate, prompt-injection attempts are flagged, and inputs are
  validated before any API call.
- **Follow-ups.** Context-dependent follow-ups ("how does *it* work?") are
  rewritten into standalone search queries.
- **Product touches.** Streaming responses, suggested starter questions, live index
  stats, clear-chat, and friendly error/rate-limit handling.
- **Eval.** `eval/run_eval.py` reports retrieval hit-rate@k and guardrail precision
  on a small gold set.

## Architecture

```
ingest.py -> scrape curated URLs -> clean+chunk -> local embeddings -> Chroma (data/chroma)
                                                                            |
app.py -- question -> embed -> Chroma top-k -> guardrail -> grounded prompt -+
 (Streamlit)           (local)                   |              |
                                         off-topic? refuse   Gemini (streaming)
                                                             |
                                         answer + [n] citations + Sources panel
```

| File | Responsibility |
|------|----------------|
| `src/config.py` | Tunables: models, paths, retrieval knobs, crawl allowlist. |
| `src/scraper.py` | Bounded BFS crawl within an allowlist; main-content extraction. |
| `src/chunker.py` | Heading-aware, token-bounded chunking with source metadata. |
| `src/vectorstore.py` | Chroma persistence + local fastembed embeddings. |
| `src/rag.py` | Follow-up rewrite, retrieve, generate (streaming). |
| `src/guardrails.py` | Input hygiene + retrieval-confidence scope/grounding gate. |
| `src/prompts.py` | System prompt + grounded context formatting. |
| `app.py` | Streamlit chat UI. |
| `ingest.py` | One-off pipeline to build the index. |
| `eval/` | Gold questions + retrieval/guardrail smoke test. |

## Running it locally

You'll need Python 3.10+ and a free Google AI Studio API key
(<https://aistudio.google.com/app/apikey>).

```bash
# 1. Clone and enter
git clone <your-repo-url> && cd gitlab-rag-chatbot

# 2. (Recommended) virtual env
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env          # then edit .env and paste your key

# 5. Build the index (one-off; takes a few minutes)
python ingest.py              # or: python ingest.py --max-pages 30  (quick test)

# 6. Launch the chatbot
streamlit run app.py
```

The vector index is committed under `data/chroma/`, so if you cloned a repo that
already has it, you can skip step 5 and go straight to `streamlit run app.py`.

### Run the eval

```bash
python eval/run_eval.py
```

## Deploying (Streamlit Community Cloud, free)

1. Push this repo to GitHub (including the committed `data/chroma/` index).
2. Go to <https://share.streamlit.io>, create a new app from your repo, main file
   `app.py`.
3. In **App -> Settings -> Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```
4. Deploy. The index ships with the repo, so the app boots without re-scraping.

## Configuration

Everything is overridable via environment variables (see `src/config.py`). The ones
you're most likely to touch:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CHAT_MODEL` | `gemini-2.5-flash` | Generation model (Gemini API). |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (fastembed). |
| `TOP_K` | `5` | Chunks retrieved per query. |
| `MIN_SIMILARITY` | `0.70` | Below this, retrieval is "weak" and the bot refuses. |
| `MAX_PAGES` | `250` | Crawl cap. |

To widen coverage, add URL prefixes to `ALLOWED_PREFIXES` / `SEED_URLS` in
`src/config.py` and re-run `python ingest.py`.

## Limitations and next steps

- It indexes a curated subset (values, engineering, people-group, company,
  leadership, product, and the live "what's new" overview) rather than the full
  handbook, mostly because of the time box. Widening it is a config change.
  Note: GitLab retired `about.gitlab.com/direction/` (it now redirects to
  `/whats-new/`), so product direction is read from `/handbook/product` instead.
- Retrieval is dense-only; hybrid (BM25 + dense) re-ranking would help on rare
  keywords.
- The handbook changes often, so a scheduled re-ingest would keep the index fresh.

See [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) for the full write-up of
the approach, decisions, and trade-offs.
