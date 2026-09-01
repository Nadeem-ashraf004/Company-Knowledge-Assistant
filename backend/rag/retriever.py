from uuid import UUID

from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.rag.embeddings import embed_query
from app.rag.vector_store import get_qdrant_client
from app.core.config import settings


def retrieve(
    query: str,
    user_id: UUID,
    top_k: int = 10,
):
    """Retrieve relevant chunks for a user."""

    query_vector = embed_query(query)

    client = get_qdrant_client()

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(
                        value=str(user_id)
                    ),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )

    return results.points