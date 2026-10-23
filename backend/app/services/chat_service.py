from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.message import Message, MessageRole


def create_conversation(
    db: Session,
    user_id: UUID,
    title: str = "New Conversation",
) -> Conversation:
    """Create a new conversation."""

    conversation = Conversation(
        user_id=user_id,
        title=title,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


def get_conversation(
    db: Session,
    conversation_id: UUID,
    user_id: UUID,
) -> Conversation | None:
    """Get a conversation belonging to a user."""

    statement = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
    )

    return db.scalar(statement)


def add_message(
    db: Session,
    conversation_id: UUID,
    user_id: UUID,
    role: MessageRole,
    content: str,
    citations: list | None = None,
    token_count: int | None = None,
) -> Message | None:
    """Add a message only to a conversation owned  by the user ."""
    conversation = get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if conversation is None:
        return None

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
        token_count=token_count,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


def get_conversation_messages(
    db: Session,
    conversation_id: UUID,
    user_id : UUID,
) -> list[Message]:
    """retrieve messages only form a conversation owned by the user."""

    statement = (
        select(Message)
        .join(
            Conversation,
            Message.conversation_id == Conversation.id,
        )
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id)
        .order_by(Message.created_at.asc())
    )

    return list(db.scalars(statement).all())