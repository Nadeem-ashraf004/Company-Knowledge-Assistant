from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus


def create_document(
    db: Session,
    user_id: UUID,
    filename: str,
    file_type: str,
    file_path: str | None = None,
    description: str | None = None,
) -> Document:
    """Create a document record."""

    document = Document(
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        description=description,
        status=DocumentStatus.PENDING,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


def get_document_by_id(
    db: Session,
    document_id: UUID,
    user_id: UUID,
) -> Document | None:
    """Get a document belonging to a specific user."""

    statement = select(Document).where(
        Document.id == document_id,
        Document.user_id == user_id,
    )

    return db.scalar(statement)


def get_user_documents(
    db: Session,
    user_id: UUID,
) -> list[Document]:
    """Get all documents belonging to a user."""

    statement = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )

    return list(db.scalars(statement).all())


def update_document_status(
    db: Session,
    document: Document,
    status: DocumentStatus,
    chunk_count: int | None = None,
) -> Document:
    """Update document processing status."""

    document.status = status

    if chunk_count is not None:
        document.chunk_count = chunk_count

    db.commit()
    db.refresh(document)

    return document


def delete_document(
    db: Session,
    document: Document,
) -> None:
    """Delete a document."""

    db.delete(document)
    db.commit()