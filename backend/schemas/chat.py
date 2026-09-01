from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_id: UUID
    document_name: str
    chunk_id: str
    page_number: int | None = None
    relevance_score: float | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: UUID
    message: str
    citations: list[Citation] = []
    created_at: datetime