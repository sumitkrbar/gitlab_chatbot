"""Chroma persistent store + local (fastembed) embeddings.

Embeddings run on a local ONNX model (BAAI/bge-small-en-v1.5 via fastembed) instead
of a hosted API, for two reasons:

- The Gemini free-tier embedding quota is tiny and the query path embeds on every
  user turn, so a rate-limited API would make the live app unreliable. Local
  embeddings are free and unlimited; the LLM API is kept for generation only.
- bge uses different prefixes for queries vs passages. fastembed's query_embed /
  passage_embed handle that, which helps recall.

Chunk ids are content hashes, so upserts are idempotent and re-running ingest.py
won't create duplicates.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

from . import config
from .chunker import Chunk

_embedder: TextEmbedding | None = None


def _model() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=config.EMBED_MODEL)
    return _embedder


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages for indexing (uses the bge passage representation)."""
    return [v.tolist() for v in _model().passage_embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single query (uses the bge query representation)."""
    return list(_model().query_embed(text))[0].tolist()


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - fine if it doesn't exist yet
            pass
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def index_chunks(chunks: list[Chunk], reset: bool = True, verbose: bool = True) -> int:
    """Embed and upsert chunks. Returns the number of vectors in the collection."""
    collection = get_collection(reset=reset)
    existing = set() if reset else set(collection.get(include=[])["ids"])
    # Dedupe within the batch too: two sections can hash to the same id (identical
    # heading + leading text), which Chroma rejects inside a single upsert.
    seen_ids: set[str] = set(existing)
    todo: list[Chunk] = []
    for c in chunks:
        if c.id in seen_ids:
            continue
        seen_ids.add(c.id)
        todo.append(c)
    if verbose:
        skipped = len(chunks) - len(todo)
        print(
            f"Embedding {len(todo)} chunks with {config.EMBED_MODEL} (local) "
            f"({skipped} already indexed) ..."
        )

    batch_size = config.EMBED_BATCH_SIZE
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        vectors = embed_documents([c.text for c in batch])
        collection.upsert(
            ids=[c.id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        if verbose:
            print(f"  indexed {min(i + batch_size, len(todo))}/{len(todo)}")
    return collection.count()


def query(text: str, top_k: int | None = None) -> list[dict]:
    """Return top-k retrieved chunks with a 0..1 similarity score."""
    top_k = top_k or config.TOP_K
    collection = get_collection()
    res = collection.query(
        query_embeddings=[embed_query(text)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
                "heading": meta.get("heading", ""),
                "similarity": round(1.0 - float(dist), 4),  # cosine distance -> sim
            }
        )
    return hits
