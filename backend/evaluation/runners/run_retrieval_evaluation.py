
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

        # Stage A: Dense retrieval from Qdrant.
        dense_documents = retrieve(
            query=query,
            user_id=TEST_USER_ID,
            top_k=RETRIEVAL_TOP_K,
        )

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
        # Stage B: Rerank the same top-10 candidates.
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

        for stage_name, metrics in (
            ("dense_top_5", dense_metrics),
            ("reranked_top_5", reranked_metrics),
        ):
            for metric_name in stages[stage_name]:
                stages[stage_name][metric_name].append(
                    metrics[metric_name]
                )

        print(f"Expected relevant chunks: {relevant_ids}")
        print(f"Dense top 5:    {dense_top_5_ids}")
        print(f"Reranked top 5: {reranked_top_5_ids}")

        print("\nDense retrieval metrics:")
        print(f"  Precision@5: {dense_metrics['precision_at_k']:.4f}")
        print(f"  Recall@5:    {dense_metrics['recall_at_k']:.4f}")
        print(f"  MRR@5:       {dense_metrics['mrr_at_k']:.4f}")

        print("\nReranked metrics:")
        print(f"  Precision@5: {reranked_metrics['precision_at_k']:.4f}")
        print(f"  Recall@5:    {reranked_metrics['recall_at_k']:.4f}")
        print(f"  MRR@5:       {reranked_metrics['mrr_at_k']:.4f}")

    print("\n" + "=" * 70)
    print("OVERALL MEAN RESULTS")
    print("=" * 70)

    for stage_name, metrics in stages.items():
        print(f"\n{stage_name.upper()}")

        for metric_name, values in metrics.items():
            label = {
                "precision_at_k": "Precision@5",
                "recall_at_k": "Recall@5",
                "mrr_at_k": "MRR@5",
            }[metric_name]

            if values:
                print(f"  Mean {label}: {mean(values):.4f}")
            else:
                print(f"  Mean {label}: N/A")



if __name__ == "__main__":
    main()

