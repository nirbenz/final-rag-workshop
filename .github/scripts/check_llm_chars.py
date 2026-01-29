#!/usr/bin/env python3
"""
Pre-commit hook that fails if staged files contain non-ASCII characters
commonly introduced by LLMs (smart quotes, em dashes, emojis, etc.).

Does NOT modify files -- report-only.
Run the companion clean_llm_chars.py to auto-fix violations.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

KNOWN_ISSUES = {"→": "->", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "·": " ", "•": " ", "×": "x"}

KNOWN_IGNORES = ["├", "─", "└", "│", "┘", "┐", "┬", "┤", "┌", "┬", "┌"]  # used by tree/visualizations
KNOWN_IGNORES += ["✓", "❌"]  # used by pytest
KNOWN_IGNORES += ["▌"]  # generally cute

NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")


def check_file(path: Path) -> list[str]:
    """
    Return a list of human-readable violation strings for *path*.

    An empty list means the file is clean.
    """
    violations: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return violations

    for lineno, line in enumerate(lines, start=1):
        for match in NON_ASCII_RE.finditer(line):
            char = match.group()
            if char in KNOWN_IGNORES:
                continue
            col = match.start() + 1
            label = KNOWN_ISSUES.get(char, f"U+{ord(char):04X}")
            violations.append(f"  {path}:{lineno}:{col}  {repr(char)} ({label})")

    return violations


def main(argv: list[str] | None = None) -> int:
    files = argv if argv else sys.argv[1:]
    if not files:
        return 0

    all_violations: list[str] = []
    for fname in files:
        all_violations.extend(check_file(Path(fname)))

    if all_violations:
        print("Non-ASCII (LLM) characters detected:\n")
        print("\n".join(all_violations))
        print(f"\n{len(all_violations)} violation(s) found.")
        print("Run .devconfig/scripts/clean_llm_chars.py to auto-fix.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
