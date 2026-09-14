from uuid import UUID

from app.rag.query_rewriter import rewrite_query
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.prompt import build_rag_prompt
from app.rag.llm import generate_answer
from app.rag.intent_detector import detect_intent


def run_rag_pipeline(
    query: str,
    user_id: UUID,
    conversation_history: list[dict] | None = None,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 5,
):
    intent = detect_intent(
    query=query,
    conversation_history=conversation_history,
    )

    if intent == "STANDALONE":
            rewritten_query = query

    elif intent == "FOLLOW_UP":
            rewritten_query = rewrite_query(
            query=query,
            conversation_history=conversation_history,
    )

    elif intent == "TOPIC_SWITCH":
            rewritten_query = query

    elif intent == "AMBIGUOUS":
           # Do not guess what the user means.
        return {
            "query": query,
            "answer": "Could you please clarify what you would like to know?",
            "documents": [],
            "intent": intent,
        }
    

    else:
       rewritten_query = query

    print("[PIPELINE] Intent:", intent)
    print("[PIPELINE] Search query:", rewritten_query)

    # 2. Retrieve relevant documents
    retrieved_documents = retrieve(
        query=rewritten_query,
        user_id=user_id,
        top_k=retrieval_top_k,
    )

    # 3. Rerank retrieved documents
    ranked_documents = rerank(
        query=rewritten_query,
        documents=retrieved_documents,
        top_k=rerank_top_k,
    )

    # 4. Build context from ranked documents
    context_parts = []

    for document in ranked_documents:
        payload = document.get("payload", {})
        text = payload.get("text", "")

        if text:
            context_parts.append(text)

    context = "\n\n".join(context_parts)

    # 5. Build the RAG prompt
    prompt = build_rag_prompt(
        question=rewritten_query,
        context=context,
    )

    # 6. Generate answer using Gemini
    answer = generate_answer(prompt)

    return {
        "query": rewritten_query,
        "answer": answer,
        "documents": ranked_documents,
    }