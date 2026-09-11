from pathlib import Path
from uuid import UUID, uuid4

from app.rag.cleaner import clean_text
from app.rag.chunker import create_chunks
from app.rag.embeddings import embed_documents
from app.rag.loader import load_document
from app.rag.vector_store import (
    create_point,
    ensure_collection,
    upsert_vectors,
)


def ingest_document(
    file_path: str,
    user_id: UUID,
    document_id: UUID | None = None,
    original_filename: str | None = None,
) -> dict:
    """
    Load, clean, chunk, embed, and store a document in Qdrant.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    raw_text = load_document(file_path)

    if not raw_text.strip():
        raise ValueError(
            "Document contains no readable text."
        )

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        raise ValueError(
            "Document contains no usable text after cleaning."
        )

    chunks = create_chunks(cleaned_text)

    if not chunks:
        raise ValueError(
            "No chunks were created from the document."
        )

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = embed_documents(texts)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    vector_size = len(embeddings[0])

    ensure_collection(vector_size)

    document_id = document_id or uuid4()

    points = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        point_id = uuid4()

        payload = {
            "user_id": str(user_id),
            "document_id": str(document_id),
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "file_name": original_filename or path.name,
            "page": chunk.page,
        }

        point = create_point(
            point_id=point_id,
            vector=embedding,
            payload=payload,
        )

        points.append(point)

    upsert_vectors(points)

    return {
        "document_id": str(document_id),
        "file_name": path.name,
        "chunks_created": len(chunks),
        "vector_size": vector_size,
    }