from app.rag.llm import generate_answer


def rewrite_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Rewrite a conversational query into a standalone search query.
    """

    query = query.strip()

    if not query:
        return query

    # No conversation means there is nothing to rewrite.
    if not conversation_history:
        return query

    history_parts = []

    # Use recent conversation context.
    for message in conversation_history[-6:]:
        role = message.get("role", "")
        content = message.get("content", "").strip()

        if content:
            history_parts.append(f"{role}: {content}")

    if not history_parts:
        return query

    history = "\n".join(history_parts)

    rewrite_prompt = f"""
You are a query rewriting component for a conversational RAG system.

Your ONLY task is to convert the latest user question into a
complete standalone search query.

Conversation:
{history}

Latest user question:
{query}

Follow these rules strictly:

1. Preserve the exact subject, entity, company, person, product,
   document, metric, or topic from the conversation when the latest
   question refers to it.

2. Resolve references such as:
   "it", "they", "this", "that", "same", "previous year",
   "next year", "last year", "how much", "how many", and
   "what about".

3. If the user asks about a previous or next year, determine the
   correct year from the conversation and include the actual year
   in the rewritten query.

4. Do NOT replace specific terms with vague terms.
   For example, do not change "Apple total net sales" into
   "results" or "performance".

5. Preserve important details such as dates, numbers, metrics,
   names, and time periods.

6. Keep the original meaning of the user's question.

7. Do NOT answer the question.

8. Do NOT explain your reasoning.

9. Return ONLY ONE standalone search query.

10. The rewritten query must contain enough information to be
    understood without the conversation.

Example:

Conversation:
user: What were Apple's total net sales for the three months ended June 27, 2026?
assistant: Apple's total net sales were $109,417 million.

Latest user question:
What about the previous year?

Correct standalone query:
What were Apple's total net sales for the three months ended June 28, 2025?

Another example:

Conversation:
user: What is Apple's revenue for 2026?
assistant: Apple's revenue was ...

Latest user question:
What about 2025?

Correct standalone query:
What was Apple's revenue for 2025?

Now rewrite the latest user question.

Return ONLY the standalone search query.
""".strip()

    rewritten_query = generate_answer(
        prompt=rewrite_prompt,
        temperature=0.0,
        max_tokens=1000,
    )

    rewritten_query = rewritten_query.strip()

    print("[QUERY REWRITER] Original:", query)
    print("[QUERY REWRITER] Rewritten:", rewritten_query)

    return rewritten_query or query