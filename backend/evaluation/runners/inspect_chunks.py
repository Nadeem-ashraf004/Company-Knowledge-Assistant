
from uuid import UUID

from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.core.config import settings
from app.rag.vector_store import get_qdrant_client


# This is the user ID associated with the test document
# from our earlier debugging session.
TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")

SEARCH_TERMS = (
    "fiscal year",
    "last Saturday of September",
    "52 or 53 week",
)


def main() -> None:
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION_NAME

    print(f"Collection: {collection_name}")
    print(f"Test user: {TEST_USER_ID}")

    offset = None
    matches = []
    page_size = 100

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=str(TEST_USER_ID)),
                    )
                ]
            ),
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            text = str(
                payload.get("text")
                or payload.get("chunk_text")
                or payload.get("content")
                or ""
            )

            if any(
                term.casefold() in text.casefold()
                for term in SEARCH_TERMS
            ):
                matches.append((record, payload, text))

        if next_offset is None:
            break

        offset = next_offset

    if not matches:
        print("\nNo matching chunks found.")
        print(
            "Check the test user ID and the text field in your "
            "Qdrant payload."
        )
        return

    print(f"\nMatching chunks found: {len(matches)}")

    for index, (record, payload, text) in enumerate(matches, start=1):
        print(f"\n{'=' * 70}")
        print(f"Match #{index}")
        print(f"Qdrant point ID: {record.id}")
        print(f"Payload chunk ID: {payload.get('chunk_id')}")
        print(f"Document ID: {payload.get('document_id')}")
        print(f"User ID: {payload.get('user_id')}")
        print(f"Page: {payload.get('page_number', payload.get('page'))}")
        print(f"Payload keys: {list(payload.keys())}")
        print(f"Text:\n{text[:1800]}")


if __name__ == "__main__":
    main()
