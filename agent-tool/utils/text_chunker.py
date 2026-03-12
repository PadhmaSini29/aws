from typing import List


def chunk_text(
    text: str,
    max_chars: int = 1800,
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks suitable for LLM prompts.

    Args:
        text: Full document text
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(0, end - overlap)

    return chunks
