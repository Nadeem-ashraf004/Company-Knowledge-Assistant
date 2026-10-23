from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user

from app.db.database import get_db
from app.models.message import MessageRole
from app.models.user import User
from app.services.chat_service import (
    add_message,
    create_conversation,
    get_conversation,
    get_conversation_messages,
)
from app.rag.pipeline import run_rag_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/")
async def chat(
    query: str,
    current_user: User = Depends(get_current_user),
    conversation_id: UUID | None = None,
    db: Session = Depends(get_db),
):
    """
    Answer a user's question using the RAG pipeline
    and persist the conversation history.
    """

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    #the authenticated user id come from the jwt
    user_id = current_user.id

    try:
        # Create a new conversation if one was not provided.
        if conversation_id is None:
            conversation = create_conversation(
                db=db,
                user_id=user_id,
                title=query[:80],
            )
        else:
            conversation = get_conversation(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

        # CHANGED:
        # Load the existing conversation history BEFORE
        # saving the current user question.
        messages = get_conversation_messages(
            db=db,
            conversation_id=conversation.id,
            user_id=user_id,
        )

        conversation_history = [
            {
                "role": (
                    message.role.value
                    if hasattr(message.role, "value")
                    else message.role
                ),
                "content": message.content,
            }
            for message in messages
        ]

        # Save the current user's message.
        saved_user_massage = add_message(
            db=db,
            conversation_id=conversation.id,
            user_id= user_id,
            role=MessageRole.USER,
            content=query,
        )
        if saved_user_massage is None:
            raise HTTPException(
                status_code= 404,
                detail="Conversation not found .",
            )
        saved_assistant_message = add_message(
            db=db,
            conversation_id=conversation.id,
            user_id=user_id,
            role=MessageRole.ASSISTANT,
            content=result["answer"],
            citations=citations,
        )

        if saved_assistant_message is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        # Run the RAG pipeline using the previous conversation history.
        result = run_rag_pipeline(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history,
        )

        # Extract citation information.
        citations = []

        for document in result["documents"]:
            payload = document.get("payload", {})

            citations.append(
                {
                    "document_id": payload.get("document_id"),
                    "file_name": payload.get("file_name"),
                    "page": payload.get("page"),
                    "chunk_id": payload.get("chunk_id"),
                    "score": document.get("rerank_score"),
                }
            )

        # Save the assistant's response.
        add_message(
            db=db,
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=result["answer"],
            citations=citations,
        )

        return {
            "conversation_id": str(conversation.id),
            "query": result["query"],
            "answer": result["answer"],
            "citations": citations,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your question.",
        )
