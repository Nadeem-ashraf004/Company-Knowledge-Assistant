from uuid import UUID

from app.rag.hybrid_search import hybrid_search
from app.rag.query_rewriter import rewrite_query
from app.rag.reranker import rerank


def run_rag_pipeline(
    query: str,
    user_id: UUID,
    conversation_history: list[dict] | None = None,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 5,
) -> dict:
    """
    Run the complete retrieval pipeline.

    Flow:
        Query
            ↓
        Query Rewriting
            ↓
        Hybrid Search
            ↓
        Cross-Encoder Reranking
            ↓
        Top-K Documents
    """

    if not query.strip():
        return {
            "query": query,
            "documents": [],
        }

    # --------------------------------------------------
    # 1. Rewrite the user's query
    # --------------------------------------------------

    rewritten_query = rewrite_query(
        query=query,
        conversation_history=conversation_history,
    )

    # --------------------------------------------------
    # 2. Hybrid retrieval
    # --------------------------------------------------

    retrieved_documents = hybrid_search(
        query=rewritten_query,
        user_id=user_id,
        top_k=retrieval_top_k,
    )

    # --------------------------------------------------
    # 3. Cross-encoder reranking
    # --------------------------------------------------

    ranked_documents = rerank(
        query=rewritten_query,
        documents=retrieved_documents,
        top_k=rerank_top_k,
    )

    # --------------------------------------------------
    # 4. Return retrieval results
    # --------------------------------------------------

    return {
        "query": rewritten_query,
        "documents": ranked_documents,
    }