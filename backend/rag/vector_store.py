from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.core.config import settings


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client."""

    if settings.QDRANT_API_KEY:
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

    return QdrantClient(
        url=settings.QDRANT_URL
    )


def ensure_collection(
    vector_size: int,
) -> None:
    """Create the Qdrant collection if it doesn't exist."""

    client = get_qdrant_client()

    collections = client.get_collections()

    collection_names = {
        collection.name
        for collection in collections.collections
    }

    if settings.QDRANT_COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def upsert_vectors(
    points: list[PointStruct],
) -> None:
    """Insert or update vectors in Qdrant."""

    client = get_qdrant_client()

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=points,
    )


def create_point(
    point_id: UUID,
    vector: list[float],
    payload: dict,
) -> PointStruct:
    """Create a Qdrant point."""

    return PointStruct(
        id=str(point_id),
        vector=vector,
        payload=payload,
    )