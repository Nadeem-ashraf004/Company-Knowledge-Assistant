
from uuid import UUID

from app.rag.embeddings import embed_query
from app.rag.vector_store import get_qdrant_client
from app.core.config import settings
from app.rag.retriever import retrieve


TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")
RELEVANT_CHUNK_ID = "chunk_20"

QUERIES = [
    "What is the company's fiscal year?",
    "When does Apple's fiscal year end?",
    "What date does Apple's fiscal year end?",
    "What is the end date of Apple's fiscal year?",
    "Apple fiscal year ending date",
    "On what day does Apple's fiscal year end?",
    "What is Apple's fiscal year-end period?",
]

def main() -> None:
    client = get_qdrant_client()

    # Find the indexed relevant chunk.
    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "user_id",
                    "match": {"value": str(TEST_USER_ID)},
                },
                {
                    "key": "chunk_id",
                    "match": {"value": RELEVANT_CHUNK_ID},
                },
            ]
        },
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        print(f"Could not find {RELEVANT_CHUNK_ID} in Qdrant.")
        return

    point = points[0]
    chunk_text = (point.payload or {}).get("text", "")
    chunk_vector = point.vector

    # Handle named-vector collections if applicable.
    if isinstance(chunk_vector, dict):
        chunk_vector = next(iter(chunk_vector.values()))

    print(f"Relevant chunk: {RELEVANT_CHUNK_ID}")
    print(f"Chunk text: {chunk_text[:500]}")

    for query in QUERIES:
        query_vector = embed_query(query)

        # Cosine similarity, assuming normalized vectors.
        dot_product = sum(
            query_value * chunk_value
            for query_value, chunk_value in zip(query_vector, chunk_vector)
        )
        results = retrieve(query, TEST_USER_ID, top_k=10)

        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if (result.payload or {}).get("chunk_id") == RELEVANT_CHUNK_ID
            ),
            None,
        )

        

        print("\n" + "-" * 70)
        print(f"Query: {query}")
        print(f"Query/chunk dot product: {dot_product:.4f}")
        print(
        "Top retrieved chunk IDs:",
        [(result.payload or {}).get("chunk_id") for result in results],
        )
        print(f"Actual retrieval rank: {rank if rank else 'Not in top 10'}")


if __name__ == "__main__":
    main()
