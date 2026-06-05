from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- API key --------------------------------------------------------------
def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your "
            "free key from https://aistudio.google.com/app/apikey"
        )
    return key

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))  # upsert chunk size

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

# --- Chunking ----------------------------------------------------------------
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "650"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

# --- Retrieval ---------------------------------------------------------------
COLLECTION_NAME = "gitlab_handbook"
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.75"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))

# --- Crawl scope ----------------------------------------------------------
# Seed from GitLab's Handbook + Direction landing pages, as in the brief.
SEED_URLS = [
    "https://about.gitlab.com/direction/",
    "https://handbook.gitlab.com/handbook/company/vision/",
    "https://handbook.gitlab.com/handbook/company/mission/",
    "https://handbook.gitlab.com/handbook/values/",
    "https://handbook.gitlab.com/handbook/engineering/",
    "https://handbook.gitlab.com/handbook/people-group/",
    "https://handbook.gitlab.com/handbook/company/",
    "https://handbook.gitlab.com/handbook/leadership/",
]

# A page is only crawled/indexed if its URL starts with one of these prefixes.
ALLOWED_PREFIXES = [
    "https://about.gitlab.com/direction",
    "https://handbook.gitlab.com/handbook/values",
    "https://handbook.gitlab.com/handbook/engineering",
    "https://handbook.gitlab.com/handbook/people-group",
    "https://handbook.gitlab.com/handbook/company",
    "https://handbook.gitlab.com/handbook/leadership",
]

MAX_PAGES = int(os.getenv("MAX_PAGES", "250"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "2"))
REQUEST_TIMEOUT = 20.0
CRAWL_WORKERS = int(os.getenv("CRAWL_WORKERS", "8"))
USER_AGENT = "gitlab-rag-chatbot/1.0 (take-home assignment; respectful crawl)"

STARTER_QUESTIONS = [
    "What are GitLab's core values?",
    "How does GitLab approach asynchronous communication?",
    "What is GitLab's product direction?",
    "How does GitLab structure its engineering organization?",
]
