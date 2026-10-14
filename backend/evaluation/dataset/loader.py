import json
from pathlib import Path
from typing import Any


# Default dataset location:
# backend/evaluation/dataset/retrieval_dataset.json
DEFAULT_DATASET_PATH = Path(__file__).with_name("retrieval_dataset.json")


REQUIRED_FIELDS = {
    "id",
    "query",
    "expected_answer",
    "relevant_chunk_ids",
}


def load_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    """
    Load and validate the retrieval evaluation dataset.

    Returns:
        A list of validated evaluation cases.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset content is invalid.
    """
    path = Path(dataset_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in dataset file: {exc}"
        ) from exc

    if not isinstance(dataset, list):
        raise ValueError(
            "Dataset must contain a JSON list of evaluation cases."
        )

    seen_ids: set[str] = set()

    for index, case in enumerate(dataset):
        location = f"Dataset case at index {index}"

        if not isinstance(case, dict):
            raise ValueError(f"{location} must be a JSON object.")

        missing_fields = REQUIRED_FIELDS - case.keys()

        if missing_fields:
            raise ValueError(
                f"{location} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        for field in ("id", "query", "expected_answer"):
            value = case[field]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{location}: '{field}' must be a non-empty string."
                )

        case_id = case["id"]

        if case_id in seen_ids:
            raise ValueError(
                f"Duplicate evaluation case ID: '{case_id}'"
            )

        seen_ids.add(case_id)

        relevant_chunk_ids = case["relevant_chunk_ids"]

        if not isinstance(relevant_chunk_ids, list):
            raise ValueError(
                f"{location}: 'relevant_chunk_ids' must be a list."
            )

        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in relevant_chunk_ids
        ):
            raise ValueError(
                f"{location}: every relevant chunk ID "
                "must be a non-empty string."
            )

        if len(relevant_chunk_ids) != len(set(relevant_chunk_ids)):
            raise ValueError(
                f"{location}: 'relevant_chunk_ids' contains duplicates."
            )

    return dataset


if __name__ == "__main__":
    try:
        cases = load_dataset()

        print("Evaluation dataset loaded successfully.")
        print(f"Total test cases: {len(cases)}")

        for case in cases:
            label_status = (
                "labeled"
                if case["relevant_chunk_ids"]
                else "needs ground-truth labels"
            )

            print(f"- {case['id']}: {label_status}")

    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset validation failed: {exc}")
        raise SystemExit(1)