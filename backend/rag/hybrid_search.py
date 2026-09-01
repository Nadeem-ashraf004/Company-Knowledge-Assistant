from uuid import UUID

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.embeddings import embed_query
from app.rag.vector_store import get_qdrant_client


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""

    return text.lower().split()


def _rrf_score(
    rank: int,
    k: int = 60,
) -> float:
    """Calculate Reciprocal Rank Fusion score."""

    return 1.0 / (k + rank)


def hybrid_search(
    query: str,
    user_id: UUID,
    top_k: int = 10,
):
    """
    Perform hybrid retrieval using:

    1. Qdrant semantic/vector search
    2. BM25 keyword search
    3. Reciprocal Rank Fusion (RRF)
    """

    if not query.strip():
        return []

    client = get_qdrant_client()

    # --------------------------------------------------
    # 1. Semantic / Vector Search
    # --------------------------------------------------

    query_vector = embed_query(query)

    vector_results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k * 2,
        with_payload=True,
    ).points

    # --------------------------------------------------
    # 2. Apply user-level filtering
    # --------------------------------------------------

    vector_results = [
        result
        for result in vector_results
        if result.payload
        and result.payload.get("user_id") == str(user_id)
    ]

    # --------------------------------------------------
    # 3. BM25 Keyword Search
    # --------------------------------------------------

    # Qdrant contains our indexed chunks. We retrieve
    # a larger candidate set and perform BM25 locally.
    keyword_results = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        limit=1000,
        with_payload=True,
    )[0]

    keyword_results = [
        point
        for point in keyword_results
        if point.payload
        and point.payload.get("user_id") == str(user_id)
        and point.payload.get("text")
    ]

    if keyword_results:
        documents = [
            point.payload["text"]
            for point in keyword_results
        ]

        tokenized_documents = [
            _tokenize(document)
            for document in documents
        ]

        bm25 = BM25Okapi(tokenized_documents)

        query_tokens = _tokenize(query)

        scores = bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k * 2]

        keyword_results = [
            keyword_results[index]
            for index in ranked_indexes
        ]

    # --------------------------------------------------
    # 4. Reciprocal Rank Fusion
    # --------------------------------------------------

    fused_results = {}

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        result_id = str(result.id)

        fused_results.setdefault(
            result_id,
            {
                "result": result,
                "score": 0.0,
            },
        )

        fused_results[result_id]["score"] += (
            _rrf_score(rank)
        )

    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):
        result_id = str(result.id)

        fused_results.setdefault(
            result_id,
            {
                "result": result,
                "score": 0.0,
            },
        )

        fused_results[result_id]["score"] += (
            _rrf_score(rank)
        )

    # --------------------------------------------------
    # 5. Sort final results
    # --------------------------------------------------

    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    # --------------------------------------------------
    # 6. Return top results
    # --------------------------------------------------

    return [
        {
            "id": item["result"].id,
            "score": item["score"],
            "payload": item["result"].payload,
        }
        for item in ranked_results[:top_k]
    ]