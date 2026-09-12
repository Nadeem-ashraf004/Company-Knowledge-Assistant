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

    # CHANGED: Do not rewrite standalone queries when there is no history.
    if not conversation_history:
        return query

    history_parts = []

    # CHANGED: Use only the most recent messages to keep the rewrite context focused.
    for message in conversation_history[-6:]:
        role = message.get("role", "")
        content = message.get("content", "").strip()

        if content:
            history_parts.append(f"{role}: {content}")

    if not history_parts:
        return query

    history = "\n".join(history_parts)

    rewrite_prompt = f"""
Rewrite the latest user question into ONE standalone search query.

Conversation:
{history}

Latest user question:
{query}

Rules:
- Resolve references such as "it", "they", "this", "that", "previous year", and "same company".
- Keep the original meaning.
- Use information from the conversation when resolving references.
- Do not answer the question.
- Do not explain anything.
- Return ONLY the rewritten search query.
- The output must be a complete question or search query.
- Never return an incomplete sentence.

Example:

Conversation:
user: What were Apple's total net sales for the three months ended June 27, 2026?
assistant: Apple's total net sales were $109,417 million.

Latest question:
What about the previous year?

Standalone query:
What were Apple's total net sales for the three months ended June 28, 2025?

Now rewrite the latest question.

Output only the standalone search query:
""".strip()

    # CHANGED: Allow enough output tokens for a complete standalone query.
    rewritten_query = generate_answer(
        prompt=rewrite_prompt,
        temperature=0.0,
        max_tokens=500,
    )

    rewritten_query = rewritten_query.strip()

    print("[QUERY REWRITER] Original:", query)
    print("[QUERY REWRITER] Rewritten:", rewritten_query)

    return rewritten_query or query