"""One-off ingestion: scrape -> chunk -> embed -> persist to Chroma.

Run this once to build the index that ships with the app:

    python ingest.py                # full curated crawl
    python ingest.py --max-pages 30 # quick smoke test

The resulting ``data/chroma`` directory is committed so the deployed app boots
instantly without re-scraping.
"""
from __future__ import annotations

import argparse
import sys

from src import config
from src.chunker import chunk_pages
from src.scraper import crawl
from src.vectorstore import index_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the GitLab RAG index.")
    parser.add_argument("--max-pages", type=int, default=config.MAX_PAGES)
    parser.add_argument("--max-depth", type=int, default=config.MAX_DEPTH)
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Upsert into the existing index instead of rebuilding from scratch.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1/3  Crawling curated GitLab pages")
    print("=" * 60)
    pages = crawl(max_pages=args.max_pages, max_depth=args.max_depth)
    if not pages:
        print("No pages scraped — check your network or the allowlist in config.py")
        return 1

    print("\n" + "=" * 60)
    print("STEP 2/3  Chunking")
    print("=" * 60)
    chunks = chunk_pages(pages)
    total_tokens = sum(c.metadata["tokens"] for c in chunks)
    print(
        f"{len(chunks)} chunks from {len(pages)} pages "
        f"(~{total_tokens:,} tokens, avg {total_tokens // max(len(chunks), 1)}/chunk)"
    )

    print("\n" + "=" * 60)
    print("STEP 3/3  Embedding + indexing")
    print("=" * 60)
    count = index_chunks(chunks, reset=not args.keep_index)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Pages scraped : {len(pages)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Collection now holds: {count} vectors")
    print(f"Persisted to : {config.CHROMA_DIR}")
    print("\nNext: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
