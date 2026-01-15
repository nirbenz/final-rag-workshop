# Workshop utility functions

from collections import defaultdict
from typing import List, Tuple


class ContextLookupError(Exception):
    """Error raised when context lookup fails."""

    pass


def get_chunk_coordinates(
    document_text: str, chunk_texts: List[str], allow_partial: bool = False
) -> List[Tuple[int, int]]:
    """
    Find the start and end coordinates of multiple chunk texts within a document.

    Uses Boyer-Moore-like approach for efficient pattern matching.

    Args:
        document_text: The full text document to search in
        chunk_texts: List of text chunks to find coordinates for
        allow_partial: If True, skip chunks not found instead of raising error

    Returns:
        List of (start_index, end_index) tuples for each found chunk

    Raises:
        ContextLookupError: If any chunk text is not found (when allow_partial=False)
    """
    if not document_text or not chunk_texts:
        return []

    def build_bad_char_table(pattern: str) -> defaultdict:
        table = defaultdict(lambda: len(pattern))
        for i, char in enumerate(pattern[:-1]):
            table[char] = len(pattern) - 1 - i
        return table

    def find_pattern(text: str, pattern: str, bad_char: defaultdict) -> int:
        n, m = len(text), len(pattern)
        if m > n:
            return -1

        skip = 0
        while skip <= n - m:
            j = m - 1

            while j >= 0 and pattern[j] == text[skip + j]:
                j -= 1

            if j < 0:
                return skip

            skip += bad_char[text[skip + m - 1]]

        return -1

    coordinates = []
    for chunk in chunk_texts:
        bad_char = build_bad_char_table(chunk)
        start_pos = find_pattern(document_text, chunk, bad_char)

        if start_pos == -1:
            if allow_partial:
                continue
            else:
                raise ContextLookupError(f"Chunk '{chunk}' not found in document")

        end_pos = start_pos + len(chunk)
        coordinates.append((start_pos, end_pos))

    return coordinates
