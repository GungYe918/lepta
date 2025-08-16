"""(MVP) 간단한 .g파일 로더"""

from __future__ import annotations
from pathlib    import Path


def load_grammar_text(path: str) -> str:
    """
    Load Grammar Text
    """
    text = Path(path).read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n")