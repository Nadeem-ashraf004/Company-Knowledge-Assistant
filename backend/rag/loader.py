from pathlib import Path

import pandas as pd
from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
}


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""

    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            pages.append(text)

    return "\n\n".join(pages)


def load_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""

    document = DocxDocument(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def load_txt(file_path: str) -> str:
    """Read a plain text file."""

    return Path(file_path).read_text(
        encoding="utf-8"
    )


def load_csv(file_path: str) -> str:
    """Convert CSV content into searchable text."""

    dataframe = pd.read_csv(file_path)

    return dataframe.to_csv(
        index=False
    )


def load_document(file_path: str) -> str:
    """Load a supported document and return its text."""

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    if extension == ".csv":
        return load_csv(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )