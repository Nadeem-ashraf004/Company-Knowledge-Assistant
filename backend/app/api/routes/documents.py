
from fastapi import APIRouter, UploadFile, File


router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    return {
        "message": "Document upload endpoint",
        "filename": file.filename,
        "content_type": file.content_type,
    }


@router.get("/")
async def get_documents():
    return {
        "message": "Document listing endpoint",
        "documents": [],
    }


@router.get("/{document_id}")
async def get_document(document_id: str):
    return {
        "message": "Document details endpoint",
        "document_id": document_id,
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    return {
        "message": "Document deletion endpoint",
        "document_id": document_id,
    }