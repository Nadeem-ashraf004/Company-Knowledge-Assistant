
from statistics import mean
from uuid import UUID

from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from evaluation.dataset.loader import load_dataset
from evaluation.evaluators.retrieval_metrics import evaluate_retrieval


# Dedicated test user's ID associated with testdata.pdf
TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")

RETRIEVAL_TOP_K = 10
EVALUATION_TOP_K = 5


def main() -> None:
    dataset = load_dataset()

    all_metrics = {
        "precision": [],
        "recall": [],
        "mrr": [],
    }

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Test cases: {len(dataset)}")
    print(f"Dense retrieval top_k: {RETRIEVAL_TOP_K}")
    print(f"Reranked evaluation top_k: {EVALUATION_TOP_K}")

    for case in dataset:
        query = case["query"]
        relevant_chunk_ids = case["relevant_chunk_ids"]

        print("\n" + "-" * 70)
        print(f"Test case: {case['id']}")
        print(f"Query: {query}")

        if not relevant_chunk_ids:
            print("SKIPPED: No ground-truth relevant chunk IDs.")
            continue

        # Stage 1: Retrieve candidates from Qdrant.
        retrieved_documents = retrieve(
            query=query,
            user_id=TEST_USER_ID,
            top_k=RETRIEVAL_TOP_K,
        )

        # Stage 2: Re-rank the retrieved candidates.
        ranked_documents = rerank(
            query=query,
            documents=retrieved_documents,
            top_k=EVALUATION_TOP_K,
        )

        # Use payload chunk IDs, not Qdrant point UUIDs.
        retrieved_chunk_ids = [
            document["payload"].get("chunk_id")
            for document in ranked_documents
            if document.get("payload", {}).get("chunk_id")
        ]

        metrics = evaluate_retrieval(
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=relevant_chunk_ids,
            k=EVALUATION_TOP_K,
        )

        print(f"Expected relevant chunks: {relevant_chunk_ids}")
        print(f"Retrieved chunk IDs: {retrieved_chunk_ids}")
        print(f"Precision@{EVALUATION_TOP_K}: {metrics['precision@k']}")
        print(f"Recall@{EVALUATION_TOP_K}: {metrics['recall@k']}")
        print(f"MRR@{EVALUATION_TOP_K}: {metrics['mrr@k']}")

        if metrics["precision@k"] is not None:
            all_metrics["precision"].append(metrics["precision@k"])
            all_metrics["recall"].append(metrics["recall@k"])
            all_metrics["mrr"].append(metrics["mrr@k"])

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    for metric_name, values in all_metrics.items():
        if values:
            print(f"Mean {metric_name}: {mean(values):.4f}")
        else:
            print(f"Mean {metric_name}: N/A (no labeled test cases)")


if __name__ == "__main__":
    main()
