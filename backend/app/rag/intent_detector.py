from app.rag.llm import generate_answer


def detect_intent(
    query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Detect whether the user's query is standalone, a follow-up,
    a topic switch, or ambiguous.
    """

    query = query.strip()

    if not query:
        return "STANDALONE"

    # CHANGED: A query without conversation history is always standalone.
    if not conversation_history:
        return "STANDALONE"

    history_parts = []

    # CHANGED: Use only recent conversation messages.
    for message in conversation_history[-6:]:
        role = message.get("role", "")
        content = message.get("content", "").strip()

        if content:
            history_parts.append(f"{role}: {content}")

    if not history_parts:
        return "STANDALONE"

    history = "\n".join(history_parts)

    prompt = f"""
Classify the latest user question.

Choose exactly ONE label:
STANDALONE
FOLLOW_UP
TOPIC_SWITCH
AMBIGUOUS

Conversation:
{history}

Latest user question:
{query}

Definitions:
- STANDALONE: Can be understood without the conversation.
- FOLLOW_UP: Depends on the previous question or answer.
- TOPIC_SWITCH: Clearly asks about a different subject.
- AMBIGUOUS: Depends on previous conversation, but the reference is unclear.

Examples:
"What about 2025?" -> FOLLOW_UP
"How many days?" -> FOLLOW_UP
"What is the company vacation policy?" -> TOPIC_SWITCH
"What is the capital of France?" -> STANDALONE
"How much?" -> AMBIGUOUS

Return ONLY the label.

Label:
""".strip()

    # CHANGED: Keep the classification output very small.
    result = generate_answer(
        prompt=prompt,
        temperature=0.0,
        max_tokens=150,
    )

    result = result.strip().upper()

    valid_intents = {
        "STANDALONE",
        "FOLLOW_UP",
        "TOPIC_SWITCH",
        "AMBIGUOUS",
    }

    if result not in valid_intents:
        print("[INTENT DETECTOR] Unexpected result:", result)
        return "STANDALONE"

    print("[INTENT DETECTOR] Query:", query)
    print("[INTENT DETECTOR] Intent:", result)

    return result