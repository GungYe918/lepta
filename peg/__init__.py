# lepta/peg/__init__.py
"""Standalone PEG submodule for lepta.

This package provides:
- AST nodes for a small PEG subset
- A PEG grammar parser (parses `%peg { ... }` block contents)
- A Packrat (memoizing) PEG engine/runtime

It is intentionally independent from lepta.lalr and other modules.
"""

from .ast import (
    Literal, CharClass, Any, Seq, Choice, Repeat, And, Not, Ref,
    RuleDef, PegGrammar,
)
from .parser import parse_peg_grammar
from .runtime import PegProgram, PegRunner
