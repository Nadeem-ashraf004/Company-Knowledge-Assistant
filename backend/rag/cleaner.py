import re


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace."""

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def remove_control_characters(text: str) -> str:
    """Remove unwanted control characters."""

    return "".join(
        character
        for character in text
        if character in "\n\t"
        or ord(character) >= 32
    )


def clean_text(text: str) -> str:
    """Run the complete text cleaning pipeline."""

    if not text:
        return ""

    text = remove_control_characters(text)

    text = normalize_whitespace(text)

    return text