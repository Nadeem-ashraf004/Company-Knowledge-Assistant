
from statistics import mean
from uuid import UUID

from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from evaluation.dataset.loader import load_dataset
from evaluation.evaluators.retrieval_metrics import evaluate_retrieval


TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")

RETRIEVAL_TOP_K = 10
EVALUATION_TOP_K = 5


def main() -> None:
    dataset = load_dataset()

    stages = {
        "dense_top_5": {
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr_at_k": [],
        },
        "reranked_top_5": {
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr_at_k": [],
        },
    }

    print("=" * 70)
    print("DENSE RETRIEVAL VS RERANKING EVALUATION")
    print("=" * 70)

    for case in dataset:
        query = case["query"]
        relevant_ids = case["relevant_chunk_ids"]

        print("\n" + "-" * 70)
        print(f"Test case: {case['id']}")
        print(f"Query: {query}")

        if not relevant_ids:
            print("SKIPPED: No ground-truth labels.")
            continue

        # Stage A: Retrieve the top 10 candidates from Qdrant.
        dense_documents = retrieve(
            query=query,
            user_id=TEST_USER_ID,
            top_k=RETRIEVAL_TOP_K,
        )

        # Evaluate the original dense top 5.
        dense_top_5_ids = [
            str(doc.payload["chunk_id"])
            for doc in dense_documents[:EVALUATION_TOP_K]
            if doc.payload and doc.payload.get("chunk_id")
        ]

        dense_metrics = evaluate_retrieval(
            retrieved_chunk_ids=dense_top_5_ids,
            relevant_chunk_ids=relevant_ids,
            k=EVALUATION_TOP_K,
        )

        # Stage B: Rerank the same top 10 candidates.
        reranked_documents = rerank(
            query=query,
            documents=dense_documents,
            top_k=EVALUATION_TOP_K,
        )

        reranked_top_5_ids = [
            str(doc["payload"]["chunk_id"])
            for doc in reranked_documents
            if doc.get("payload", {}).get("chunk_id")
        ]

        reranked_metrics = evaluate_retrieval(
            retrieved_chunk_ids=reranked_top_5_ids,
            relevant_chunk_ids=relevant_ids,
            k=EVALUATION_TOP_K,
        )

        # Store metrics for the final aggregate summary.
        for stage_name, metrics in (
            ("dense_top_5", dense_metrics),
            ("reranked_top_5", reranked_metrics),
        ):
            for metric_name in stages[stage_name]:
                value = metrics[metric_name]
                if value is not None:
                    stages[stage_name][metric_name].append(value)

        print(f"Expected relevant chunks: {relevant_ids}")

        # Display every dense candidate, up to 10.
        print("\nDense ranking (all retrieved candidates):")
        for rank, doc in enumerate(dense_documents, start=1):
            payload = doc.payload or {}
            print(
                f"  {rank}. chunk_id={payload.get('chunk_id')} "
                f"dense_score={doc.score:.4f}"
            )

        # Display the reranked top 5.
        print("\nReranked ranking:")
        for rank, doc in enumerate(reranked_documents, start=1):
            payload = doc.get("payload", {})
            print(
                f"  {rank}. chunk_id={payload.get('chunk_id')} "
                f"dense_score={doc.get('score', 0):.4f} "
                f"rerank_score={doc.get('rerank_score', 0):.4f}"
            )

        print("\nDense retrieval metrics:")
        print(
            f"  Precision@5: "
            f"{dense_metrics['precision_at_k']:.4f}"
        )
        print(
            f"  Recall@5:    "
            f"{dense_metrics['recall_at_k']:.4f}"
        )
        print(
            f"  MRR@5:       "
            f"{dense_metrics['mrr_at_k']:.4f}"
        )

        print("\nReranked metrics:")
        print(
            f"  Precision@5: "
            f"{reranked_metrics['precision_at_k']:.4f}"
        )
        print(
            f"  Recall@5:    "
            f"{reranked_metrics['recall_at_k']:.4f}"
        )
        print(
            f"  MRR@5:       "
            f"{reranked_metrics['mrr_at_k']:.4f}"
        )

    # This summary must be outside the test-case loop.
    print("\n" + "=" * 70)
    print("OVERALL MEAN RESULTS")
    print("=" * 70)

    labels = {
        "precision_at_k": "Precision@5",
        "recall_at_k": "Recall@5",
        "mrr_at_k": "MRR@5",
    }

    for stage_name, metrics in stages.items():
        print(f"\n{stage_name.upper()}")

        for metric_name, values in metrics.items():
            label = labels[metric_name]

            if values:
                print(f"  Mean {label}: {mean(values):.4f}")
            else:
                print(f"  Mean {label}: N/A")

if __name__ == "__main__":
    main()
