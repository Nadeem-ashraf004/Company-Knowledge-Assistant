from uuid import UUID

from app.core.config import settings
from app.rag.vector_store import get_qdrant_client

TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")

CHUNK_IDS = [
    "chunk_20",
    "chunk_50",
    "chunk_138",
    "chunk_36",
    "chunk_21",
]


def main() -> None:
    client = get_qdrant_client()

    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "user_id",
                    "match": {"value": str(TEST_USER_ID)},
                },
            ]
        },
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")

        if chunk_id not in CHUNK_IDS:
            continue

        print("\n" + "=" * 80)
        print(f"Chunk ID: {chunk_id}")
        print(f"Page: {payload.get('page')}")
        print(payload.get("text", "")[:1500])


if __name__ == "__main__":
    main()