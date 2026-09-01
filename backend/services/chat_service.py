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
    role: MessageRole,
    content: str,
    citations: list | None = None,
    token_count: int | None = None,
) -> Message:
    """Add a message to a conversation."""

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
) -> list[Message]:
    """Get messages for a conversation."""

    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )

    return list(db.scalars(statement).all())