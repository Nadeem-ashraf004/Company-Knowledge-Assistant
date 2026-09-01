def rewrite_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Rewrite a conversational query into a standalone query.

    LLM-based rewriting will be implemented later.
    """

    return query.strip()