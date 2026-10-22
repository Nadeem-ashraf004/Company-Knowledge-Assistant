
from uuid import UUID

from app.core.config import settings
from app.rag.retriever import retrieve
from app.rag.reranker import rerank ,get_reranker

TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")

QUERIES = [
    "What is the company's fiscal year?",
    "When does Apple's fiscal year end?",
]

RETRIEVAL_TOP_K = 30
RERANK_TOP_K = 10
def inspect_token_length(query: str, text: str) -> None:
    model = get_reranker()
    tokenizer = model.tokenizer

    full_tokens = tokenizer(
        query,
        text,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]

    truncated_tokens = tokenizer(
        query,
        text,
        truncation=True,
        max_length=model.max_seq_length,
        add_special_tokens=True,
    )["input_ids"]

    print(f"  Full pair tokens: {len(full_tokens)}")
    print(f"  Tokens after truncation: {len(truncated_tokens)}")
    print(f"  Truncated: {len(full_tokens) > len(truncated_tokens)}")

def main() -> None:
    for query in QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        dense_documents = retrieve(
            query=query,
            user_id=TEST_USER_ID,
            top_k=RETRIEVAL_TOP_K,
        )

        print("\n--- DENSE RETRIEVAL ---")

        for rank, document in enumerate(dense_documents, start=1):
            payload = document.payload or {}

            print(
                f"Rank: {rank} | "
                f"Chunk: {payload.get('chunk_id')} | "
                f"Dense score: {document.score}"
            )

        reranked_documents = rerank(
            query=query,
            documents=dense_documents,
            top_k=RERANK_TOP_K,
        )

        print("\n--- CROSS-ENCODER RERANKING ---")

        for rank, document in enumerate(reranked_documents, start=1):
            payload = document.get("payload", {})

            print(
                f"Rank: {rank} | "
                f"Chunk: {payload.get('chunk_id')} | "
                f"Dense score: {document.get('score')} | "
                f"Rerank score: {document.get('rerank_score')}"
            )

            if payload.get("chunk_id") == "chunk_20":
                print("  >>> THIS IS THE RELEVANT CHUNK")

            print(f"  Text: {payload.get('text', '')[:500]}")



if __name__ == "__main__":
    main()
