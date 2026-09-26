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

router = APIRouter()    

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    current_user : User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a document, store its metadata, and ingest it into Qdrant.
    """

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

    try:
        # Create a PostgreSQL document record.
        document = create_document(
            db=db,
            user_id=user_id,
            filename=file.filename,
            file_type=extension,
            description=description,
        )

        # Create a user-specific upload directory.
        user_upload_dir = UPLOAD_DIR / str(user_id)
        user_upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = user_upload_dir / f"{document.id}{extension}"

        # Save uploaded file.
        contents = await file.read()
        file_path.write_bytes(contents)

        # Store the file path.
        document.file_path = str(file_path)
        db.commit()
        db.refresh(document)

        # Mark document as processing.
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
            original_filename=file.filename,
        )

        # Mark document as completed.
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
            "status":(
                 document.status.value 
                 if hasattr(document.status,"value") 
                 else document.status
                 ),
            "chunks_created": result["chunks_created"],
        }

    except HTTPException:
        raise

    except Exception as exc:
        if "document" in locals():
            update_document_status(
                db=db,
                document=document,
                status=DocumentStatus.FAILED,
            )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.get("/")
async def list_documents(
    current_user:  User = Depends(get_current_user) ,
    db: Session = Depends(get_db),
):
    """List all documents belonging to a user."""

    documents = get_user_documents(
        db=db,
        user_id=current_user.id,
    )

    return [
        {
            "document_id": str(document.id),
            "filename": document.filename,
            "file_type": document.file_type,
            "status": document.status.value,
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
    """Get details of a user's document."""

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
        "status": document.status.value,
        "chunk_count": document.chunk_count,
        "description": document.description,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }