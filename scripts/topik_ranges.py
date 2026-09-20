"""Conservative parsing for bracketed TOPIK question ranges."""

from __future__ import annotations

import re


_BRACKETED_RANGE_RE = re.compile(
    r"\[\s*([0-9Il|OoC]{1,2})[ \t]*([;~～\-– \t]+)"
    r"([0-9Il|OoC]{1,2})\s*\]"
)
_RANGE_MARKS = frozenset("~～-–")
_OCR_DIGITS = str.maketrans(
    {"I": "1", "l": "1", "|": "1", "O": "0", "o": "0", "C": "6"}
)


def _normalize_range_endpoint(token: str) -> int | None:
    normalized = token.translate(_OCR_DIGITS)
    if not normalized.isdigit():
        return None
    number = int(normalized)
    return number if 1 <= number <= 50 else None


def parse_instruction_range(text: str) -> tuple[int, int] | None:
    """Parse one valid bracketed TOPIK range without changing surrounding OCR text."""
    parsed: list[tuple[int, int]] = []
    for match in _BRACKETED_RANGE_RE.finditer(text or ""):
        if not any(mark in match.group(2) for mark in _RANGE_MARKS):
            continue
        start = _normalize_range_endpoint(match.group(1))
        end = _normalize_range_endpoint(match.group(3))
        if start is not None and end is not None and start <= end:
            parsed.append((start, end))

    # Multiple different ranges in one OCR block are ambiguous.
    unique = set(parsed)
    return parsed[0] if len(unique) == 1 else None
