# lepta/peg/ast.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

# ---- PEG AST node definitions ----

@dataclass(frozen=True)
class Literal:
    text: str  # unescaped text

@dataclass(frozen=True)
class CharClass:
    negated: bool
    # ranges are inclusive (lo..hi). singles is a list of single codepoints (as str of length 1)
    ranges: List[Tuple[int, int]] = field(default_factory=list)
    singles: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Any:
    pass

@dataclass(frozen=True)
class Ref:
    name: str

@dataclass(frozen=True)
class And:
    node: "Node"  # positive lookahead (&)

@dataclass(frozen=True)
class Not:
    node: "Node"  # negative lookahead (!)

@dataclass(frozen=True)
class Repeat:
    node: "Node"
    kind: str  # '?', '*', '+'

@dataclass(frozen=True)
class Seq:
    items: List["Node"]

@dataclass(frozen=True)
class Choice:
    alts: List["Node"]

Node = Union[Literal, CharClass, Any, Ref, And, Not, Repeat, Seq, Choice]

@dataclass
class RuleDef:
    name: str
    expr: Node

@dataclass
class PegGrammar:
    rules: Dict[str, RuleDef]
    start: str

    def require_rule(self, name: str) -> RuleDef:
        try:
            return self.rules[name]
        except KeyError:
            raise SyntaxError(f"PEG: undefined rule '{name}'")
