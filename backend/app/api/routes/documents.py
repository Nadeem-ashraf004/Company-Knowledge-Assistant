import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.document import DocumentStatus
from app.models.user import User
from app.rag.ingestion import ingest_document
from app.services.document_service import (
    create_document,
    get_document,
    get_user_documents,
    update_document_status,
)

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    description: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a document, store its metadata, and ingest it into Qdrant."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    user_id = current_user.id
    document = None
    file_path = None

    try:
        # Create the database record.
        document = create_document(
            db=db,
            user_id=user_id,
            filename=Path(file.filename).name,
            file_type=extension,
            description=description,
        )

        # Create a user-specific upload directory.
        user_upload_dir = UPLOAD_DIR / str(user_id)
        user_upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = user_upload_dir / f"{document.id}{extension}"

        # Save the file in chunks to avoid reading the entire
        # upload into memory at once.
        total_size = 0

        with file_path.open("wb") as destination:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File is too large. Maximum upload size is 25 MB.",
                    )

                destination.write(chunk)

        # Save the file path in the database.
        document.file_path = str(file_path)
        db.commit()
        db.refresh(document)

        # Mark the document as processing.
        update_document_status(
            db=db,
            document=document,
            status=DocumentStatus.PROCESSING,
        )

        # Run RAG ingestion.
        result = ingest_document(
            file_path=str(file_path),
            user_id=user_id,
            document_id=document.id,
            original_filename=Path(file.filename).name,
        )

        # Mark the document as completed.
        update_document_status(
            db=db,
            document=document,
            status=DocumentStatus.COMPLETED,
            chunk_count=result["chunks_created"],
        )

        return {
            "document_id": str(document.id),
            "filename": document.filename,
            "file_type": document.file_type,
            "status": (
                document.status.value
                if hasattr(document.status, "value")
                else document.status
            ),
            "chunks_created": result["chunks_created"],
        }

    except HTTPException:
        # Clean up partial files and mark any created record as failed.
        if file_path is not None:
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Failed to remove incomplete upload")

        if document is not None:
            try:
                update_document_status(
                    db=db,
                    document=document,
                    status=DocumentStatus.FAILED,
                )
            except Exception:
                logger.exception("Failed to update document status")

        raise

    except Exception:
        logger.exception("Document upload or ingestion failed")

        if file_path is not None:
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Failed to remove uploaded file after error")

        if document is not None:
            try:
                update_document_status(
                    db=db,
                    document=document,
                    status=DocumentStatus.FAILED,
                )
            except Exception:
                logger.exception("Failed to update document status")

        raise HTTPException(
            status_code=500,
            detail="Document upload or processing failed. Please try again.",
        )

    finally:
        await file.close()


@router.get("/")
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all documents belonging to the current user."""

    documents = get_user_documents(
        db=db,
        user_id=current_user.id,
    )

    return [
        {
            "document_id": str(document.id),
            "filename": document.filename,
            "file_type": document.file_type,
            "status": (
                document.status.value
                if hasattr(document.status, "value")
                else document.status
            ),
            "chunk_count": document.chunk_count,
            "created_at": document.created_at,
        }
        for document in documents
    ]


@router.get("/{document_id}")
async def get_document_details(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get details of a document owned by the current user."""

    document = get_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "file_type": document.file_type,
        "status": (
            document.status.value
            if hasattr(document.status, "value")
            else document.status
        ),
        "chunk_count": document.chunk_count,
        "description": document.description,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }