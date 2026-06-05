# GitLab Handbook Assistant

A RAG chatbot that answers questions about how GitLab works, using its public
[Handbook](https://handbook.gitlab.com/) and Direction pages as the source of truth.

Work in progress (take-home assignment).

## Setup

- Python 3.10+
- `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and add a free Google AI Studio API key.

## Plan

- Scrape a curated set of handbook pages
- Chunk + embed into a local Chroma index
- Answer questions with Gemini over the retrieved chunks, with citations
