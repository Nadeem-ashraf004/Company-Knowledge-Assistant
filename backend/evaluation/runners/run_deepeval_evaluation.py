
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from app.core.config import settings
from app.rag.pipeline import run_rag_pipeline
from evaluation.dataset.loader import load_dataset


TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")

REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "deepeval_evaluation.json"
)


def main() -> None:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")

    judge_model = GeminiModel(
        model=settings.LLM_MODEL,
        api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )

    metrics = [
        FaithfulnessMetric(
            threshold=0.7,
            model=judge_model,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=0.7,
            model=judge_model,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=0.7,
            model=judge_model,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=0.7,
            model=judge_model,
            include_reason=True,
        ),
    ]

    dataset = load_dataset()
    case_reports = []

    print("=" * 70)
    print("DEEPEVAL RAG EVALUATION")
    print("=" * 70)

    for case in dataset:
        if not case.get("expected_answer"):
            print(f"Skipping {case['id']}: missing expected_answer.")
            continue

        print(f"\nCase: {case['id']}")
        print(f"Question: {case['query']}")

        # Run the actual application pipeline.
        result = run_rag_pipeline(
            query=case["query"],
            user_id=TEST_USER_ID,
        )

        # Extract the final reranked context used by the pipeline.
        retrieval_context = [
            document.get("payload", {}).get("text", "")
            for document in result.get("documents", [])
            if document.get("payload", {}).get("text")
        ]

        test_case = LLMTestCase(
            input=case["query"],
            actual_output=result["answer"],
            expected_output=case["expected_answer"],
            retrieval_context=retrieval_context,
        )

        case_report = {
            "id": case["id"],
            "input": case["query"],
            "actual_output": result["answer"],
            "expected_output": case["expected_answer"],
            "retrieval_context_count": len(retrieval_context),
            "metrics": {},
        }

        for metric in metrics:
            metric.measure(test_case)

            score = metric.score
            reason = metric.reason
            passed = metric.is_successful()

            case_report["metrics"][metric.__class__.__name__] = {
                "score": score,
                "threshold": metric.threshold,
                "passed": passed,
                "reason": reason,
            }

            print(
                f"{metric.__class__.__name__}: "
                f"score={score}, passed={passed}"
            )

            if reason:
                print(f"  Reason: {reason}")

        case_reports.append(case_report)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": settings.LLM_MODEL,
        "total_cases": len(dataset),
        "evaluated_cases": len(case_reports),
        "metrics": [
            "FaithfulnessMetric",
            "AnswerRelevancyMetric",
            "ContextualPrecisionMetric",
            "ContextualRecallMetric",
        ],
        "cases": case_reports,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print(f"Evaluated cases: {len(case_reports)}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
