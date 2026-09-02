from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""

    return SentenceTransformer(
        settings.EMBEDDING_MODEL
    )


def embed_documents(
    texts: list[str],
) -> list[list[float]]:
    """Generate embeddings for documents."""

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(
    query: str,
) -> list[float]:
    """Generate an embedding for a query."""

    model = get_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    return embedding.tolist()