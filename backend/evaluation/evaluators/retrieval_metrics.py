
from typing import Any


def evaluate_retrieval(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int = 5,
) -> dict[str, float | int | None]:
    """
    Calculate Precision@K, Recall@K, and Reciprocal Rank@K.

    Args:
        retrieved_chunk_ids: Chunk IDs returned by the retriever,
            ordered from most relevant to least relevant.
        relevant_chunk_ids: Verified ground-truth relevant chunk IDs.
        k: Number of top retrieved chunks to evaluate.

    Returns:
        A dictionary containing the metrics.

        If no ground-truth labels are provided, metrics are None
        because meaningful evaluation is not possible yet.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not isinstance(retrieved_chunk_ids, list):
        raise TypeError("retrieved_chunk_ids must be a list.")

    if not isinstance(relevant_chunk_ids, list):
        raise TypeError("relevant_chunk_ids must be a list.")

    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip()
        for chunk_id in retrieved_chunk_ids + relevant_chunk_ids
    ):
        raise ValueError("All chunk IDs must be non-empty strings.")

    # Without ground-truth labels, we cannot determine relevance.
    if not relevant_chunk_ids:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr_at_k": None,
            "k": k,
            "labeled": False,
        }

    top_k_ids = retrieved_chunk_ids[:k]
    relevant_set = set(relevant_chunk_ids)

    # Count unique relevant chunks in the retrieved results.
    relevant_retrieved = len(set(top_k_ids) & relevant_set)

    precision = relevant_retrieved / k
    recall = relevant_retrieved / len(relevant_set)

    # Reciprocal rank is based on the first relevant result.
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(top_k_ids, start=1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr_at_k": reciprocal_rank,
        "k": k,
        "labeled": True,
    }


def evaluate_dataset_results(
    results: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """
    Aggregate retrieval metrics across evaluation cases.

    Each result must contain:
        - id
        - retrieved_chunk_ids
        - relevant_chunk_ids
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    case_results = []
    labeled_results = []

    for case in results:
        for field in (
            "id",
            "retrieved_chunk_ids",
            "relevant_chunk_ids",
        ):
            if field not in case:
                raise ValueError(
                    f"Evaluation result is missing field: {field}"
                )

        metrics = evaluate_retrieval(
            retrieved_chunk_ids=case["retrieved_chunk_ids"],
            relevant_chunk_ids=case["relevant_chunk_ids"],
            k=k,
        )

        case_result = {
            "id": case["id"],
            **metrics,
        }
        case_results.append(case_result)

        if metrics["labeled"]:
            labeled_results.append(metrics)

    def average(metric_name: str) -> float | None:
        if not labeled_results:
            return None

        return sum(
            result[metric_name] for result in labeled_results
        ) / len(labeled_results)

    return {
        "k": k,
        "total_cases": len(results),
        "labeled_cases": len(labeled_results),
        "unlabeled_cases": len(results) - len(labeled_results),
        "mean_precision_at_k": average("precision_at_k"),
        "mean_recall_at_k": average("recall_at_k"),
        "mean_mrr_at_k": average("mrr_at_k"),
        "case_results": case_results,
    }


if __name__ == "__main__":
    # Small example with known ground-truth labels.
    demo = evaluate_retrieval(
        retrieved_chunk_ids=["chunk_A", "chunk_B", "chunk_C"],
        relevant_chunk_ids=["chunk_B", "chunk_D"],
        k=3,
    )

    print("Example retrieval metrics:")
    for metric, value in demo.items():
        print(f"{metric}: {value}")

    # Demonstrate how unlabeled questions are handled.
    unlabeled = evaluate_retrieval(
        retrieved_chunk_ids=["chunk_A", "chunk_B"],
        relevant_chunk_ids=[],
        k=2,
    )

    print("\nUnlabeled example:")
    for metric, value in unlabeled.items():
        print(f"{metric}: {value}")
