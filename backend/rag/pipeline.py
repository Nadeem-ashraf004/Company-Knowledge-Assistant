from uuid import UUID

from app.rag.query_rewriter import rewrite_query
from app.rag.retriever import retrieve
from app.rag.reranker import rerank


def run_rag_pipeline(
    query: str,
    user_id: UUID,
    conversation_history: list[dict] | None = None,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 5,
):
    """
    Run the complete RAG pipeline.

    Current flow:
    Query rewriting
        ↓
    Vector retrieval
        ↓
    Reranking

    LLM generation will be connected later.
    """

    rewritten_query = rewrite_query(
        query=query,
        conversation_history=conversation_history,
    )

    retrieved_documents = retrieve(
        query=rewritten_query,
        user_id=user_id,
        top_k=retrieval_top_k,
    )

    ranked_documents = rerank(
        query=rewritten_query,
        documents=retrieved_documents,
        top_k=rerank_top_k,
    )

    return {
        "query": rewritten_query,
        "documents": ranked_documents,
    }