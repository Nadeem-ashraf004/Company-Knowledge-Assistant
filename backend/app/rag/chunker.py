import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    chunk_index: int
    page: int | None = None


def create_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    """
    Split text into chunks while preserving PDF page metadata.
    """

    # Detect page markers created by loader.py
    page_pattern = re.compile(
        r"\[\[PAGE:(\d+)\]\]"
    )

    matches = list(page_pattern.finditer(text))

    # Non-PDF documents have no page markers.
    if not matches:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        chunks = splitter.split_text(text)

        return [
            TextChunk(
                chunk_id=f"chunk_{index}",
                text=chunk,
                chunk_index=index,
                page=None,
            )
            for index, chunk in enumerate(chunks)
        ]

    # Build page-aware sections.
    page_sections = []

    for index, match in enumerate(matches):
        page_number = int(match.group(1))

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        page_text = text[start:end].strip()

        if page_text:
            page_sections.append(
                (
                    page_number,
                    page_text,
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    result = []

    chunk_index = 0

    for page_number, page_text in page_sections:
        chunks = splitter.split_text(page_text)

        for chunk in chunks:
            result.append(
                TextChunk(
                    chunk_id=f"chunk_{chunk_index}",
                    text=chunk,
                    chunk_index=chunk_index,
                    page=page_number,
                )
            )

            chunk_index += 1

    return result