
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from evaluation.dataset.loader import load_dataset
from evaluation.evaluators.retrieval_metrics import (
    evaluate_retrieval,
    evaluate_dataset_results,
)


TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")
RETRIEVAL_TOP_K = 10
EVALUATION_K = 5

REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "retrieval_evaluation.json"
)


def get_chunk_ids(documents: list) -> list[str]:
    """Extract stable chunk IDs from Qdrant payloads or reranker results."""
    chunk_ids = []

    for document in documents:
        if isinstance(document, dict):
            payload = document.get("payload") or {}
        else:
            payload = document.payload or {}

        chunk_id = payload.get("chunk_id")

        if chunk_id:
            chunk_ids.append(str(chunk_id))

    return chunk_ids


def main() -> None:
    dataset = load_dataset()
    case_reports = []

    print("=" * 70)
    print("DENSE RETRIEVAL VS RERANKING EVALUATION")
    print("=" * 70)

    for case in dataset:
        query = case["query"]
        relevant_chunk_ids = case.get("relevant_chunk_ids", [])

        print(f"\nCase: {case['id']}")
        print(f"Query: {query}")
        print(f"Expected relevant chunks: {relevant_chunk_ids}")

        # Stage 1: Dense retrieval
        dense_documents = retrieve(
            query=query,
            user_id=TEST_USER_ID,
            top_k=RETRIEVAL_TOP_K,
        )

        dense_chunk_ids = get_chunk_ids(dense_documents)

        # Stage 2: Cross-encoder reranking
        reranked_documents = rerank(
            query=query,
            documents=dense_documents,
            top_k=EVALUATION_K,
        )

        reranked_chunk_ids = get_chunk_ids(reranked_documents)

        dense_metrics = evaluate_retrieval(
            retrieved_chunk_ids=dense_chunk_ids,
            relevant_chunk_ids=relevant_chunk_ids,
            k=EVALUATION_K,
        )

        reranked_metrics = evaluate_retrieval(
            retrieved_chunk_ids=reranked_chunk_ids,
            relevant_chunk_ids=relevant_chunk_ids,
            k=EVALUATION_K,
        )

        print(f"Dense top {EVALUATION_K}: {dense_chunk_ids[:EVALUATION_K]}")
        print(f"Reranked top {EVALUATION_K}: {reranked_chunk_ids}")
        print(
            "Dense metrics: "
            f"P@{EVALUATION_K}={dense_metrics['precision_at_k']}, "
            f"R@{EVALUATION_K}={dense_metrics['recall_at_k']}, "
            f"MRR@{EVALUATION_K}={dense_metrics['mrr_at_k']}"
        )
        print(
            "Reranked metrics: "
            f"P@{EVALUATION_K}={reranked_metrics['precision_at_k']}, "
            f"R@{EVALUATION_K}={reranked_metrics['recall_at_k']}, "
            f"MRR@{EVALUATION_K}={reranked_metrics['mrr_at_k']}"
        )

        case_reports.append(
            {
                "id": case["id"],
                "query": query,
                "relevant_chunk_ids": relevant_chunk_ids,
                "dense": {
                    "retrieved_chunk_ids": dense_chunk_ids,
                    **dense_metrics,
                },
                "reranked": {
                    "retrieved_chunk_ids": reranked_chunk_ids,
                    **reranked_metrics,
                },
            }
        )

    dense_aggregate = evaluate_dataset_results(
        [
            {
                "id": case["id"],
                "retrieved_chunk_ids": case["dense"]["retrieved_chunk_ids"],
                "relevant_chunk_ids": case["relevant_chunk_ids"],
            }
            for case in case_reports
        ],
        k=EVALUATION_K,
    )

    reranked_aggregate = evaluate_dataset_results(
        [
            {
                "id": case["id"],
                "retrieved_chunk_ids": case["reranked"]["retrieved_chunk_ids"],
                "relevant_chunk_ids": case["relevant_chunk_ids"],
            }
            for case in case_reports
        ],
        k=EVALUATION_K,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_k": EVALUATION_K,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "total_cases": len(dataset),
        "dense_aggregate": dense_aggregate,
        "reranked_aggregate": reranked_aggregate,
        "cases": case_reports,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"Dense retrieval: {dense_aggregate}")
    print(f"Reranked retrieval: {reranked_aggregate}")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
