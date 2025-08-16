# lepta/peg/runtime.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
from .ast import PegGrammar
from .parser import parse_peg_grammar
from .engine import Packrat

@dataclass
class PegProgram:
    """Compiled PEG program."""
    grammar: PegGrammar

    @classmethod
    def from_source(cls, src: str) -> "PegProgram":
        g = parse_peg_grammar(src)
        return cls(g)

class PegRunner:
    """Execute PEG program on input text at a given position."""
    def __init__(self, program: PegProgram):
        self.program = program

    def run(self, rule_name: str, text: str, pos: int = 0) -> Tuple[bool, int]:
        engine = Packrat(self.program.grammar)
        return engine.parse(rule_name, text, pos)
