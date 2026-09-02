from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    chunk_index: int


def create_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    """Split document text into overlapping chunks."""

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
        )
        for index, chunk in enumerate(chunks)
    ]