from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings


@lru_cache
def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder reranking model."""

    return CrossEncoder(
        settings.RERANKER_MODEL
    )


def rerank(
    query: str,
    documents: list,
    top_k: int = 5,
) -> list:
    """
    Re-rank retrieved documents using a cross-encoder.

    Args:
        query: User's search query.
        documents: Documents returned by hybrid search.
        top_k: Number of documents to return.

    Returns:
        Top-ranked documents as dictionaries.
    """

    if not query.strip() or not documents:
        return []

    # Extract document text
    pairs = []

    for document in documents:
        payload = document.payload or {}

        text = payload.get("text", "")

        pairs.append(
            (query, text)
        )

    if not pairs:
        return []

    # Generate relevance scores
    model = get_reranker()

    scores = model.predict(
        pairs
    )

    # Convert Qdrant documents into dictionaries
    ranked_documents = []

    for document, score in zip(
        documents,
        scores,
    ):
        payload = document.payload or {}

        ranked_document = {
            "id": str(document.id),
            "score": float(document.score),
            "rerank_score": float(score),
            "payload": payload,
        }

        ranked_documents.append(
            ranked_document
        )

    # Sort by reranker score
    ranked_documents.sort(
        key=lambda document: document["rerank_score"],
        reverse=True,
    )

    return ranked_documents[:top_k]