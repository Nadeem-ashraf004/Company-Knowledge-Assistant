
from uuid import UUID

from app.rag.retriever import retrieve


TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")

QUERIES = [
    "What is the company's fiscal year?",
    "When does Apple's fiscal year end?",
]


def main() -> None:
    for query in QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        documents = retrieve(
            query=query,
            user_id=TEST_USER_ID,
            top_k=10,
        )

        for rank, document in enumerate(documents, start=1):
            payload = document.payload or {}

            print(f"\nRank: {rank}")
            print(f"Chunk ID: {payload.get('chunk_id')}")
            print(f"Dense score: {document.score:.4f}")
            print(f"Text: {payload.get('text', '')[:1000]}")
            print("-" * 80)


if __name__ == "__main__":
    main()
