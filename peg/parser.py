# lepta/peg/parser.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict
from .ast import (
    Literal, CharClass, Any, Ref, And, Not, Repeat, Seq, Choice,
    RuleDef, PegGrammar, Node
)

# Grammar (PEG subset) we parse:
#   grammar  := (rule)*
#   rule     := IDENT "<-" expr
#   expr     := seq ("/" seq)*
#   seq      := (prefix)*
#   prefix   := ("&"|"!")? suffix
#   suffix   := primary ("?"|"*"|"+")?
#   primary  := IDENT | literal | class | "." | "(" expr ")"
#
#   literal  := ' ... ' | " ... "  (supports escapes \n \r \t \\ \" \' \xHH \uXXXX)
#   class    := "[" "^"? class_items "]"
#   class_items: (range | escaped | raw_char)+
#   range    := char "-" char
#   comments/space allowed:
#       - whitespace
#       - "#" ... endline
#       - "//" ... endline
#       - "/*" ... "*/"

class _TS:
    def __init__(self, src: str):
        self.s = src
        self.i = 0
        self.n = len(src)

    def _peek(self, k: int = 0) -> Optional[str]:
        j = self.i + k
        if j >= self.n:
            return None
        return self.s[j]

    def _starts(self, lit: str) -> bool:
        return self.s.startswith(lit, self.i)

    def _bump(self, n: int = 1) -> None:
        self.i += n

    def _eof(self) -> bool:
        return self.i >= self.n

    def _err(self, msg: str) -> SyntaxError:
        return SyntaxError(f"PEG parse error at {self.i}: {msg}")

    def _skip_ws(self) -> None:
        while not self._eof():
            if self._starts("/*"):
                self._bump(2)
                j = self.s.find("*/", self.i)
                if j == -1:
                    raise self._err("unclosed block comment")
                self.i = j + 2
                continue
            ch = self._peek()
            if ch in " \t\r\n":
                self._bump(1)
                continue
            if self._starts("//"):
                self._bump(2)
                while not self._eof() and self._peek() not in ("\n",):
                    self._bump(1)
                continue
            if ch == "#":
                while not self._eof() and self._peek() not in ("\n",):
                    self._bump(1)
                continue
            break

    def _eat(self, lit: str) -> None:
        self._skip_ws()
        if not self._starts(lit):
            raise self._err(f"expected {lit!r}")
        self._bump(len(lit))

    def _try_eat(self, lit: str) -> bool:
        self._skip_ws()
        if self._starts(lit):
            self._bump(len(lit))
            return True
        return False

    def _is_ident_start(self, ch: Optional[str]) -> bool:
        if ch is None:
            return False
        return ch.isalpha() or ch == "_"

    def _is_ident_continue(self, ch: Optional[str]) -> bool:
        if ch is None:
            return False
        return ch.isalnum() or ch == "_"

    def _ident(self) -> str:
        self._skip_ws()
        ch = self._peek()
        if not self._is_ident_start(ch):
            raise self._err("expected IDENT")
        start = self.i
        self._bump(1)
        while self._is_ident_continue(self._peek()):
            self._bump(1)
        return self.s[start:self.i]

    def _hexval(self, ch: str) -> int:
        if "0" <= ch <= "9": return ord(ch) - ord("0")
        if "a" <= ch <= "f": return ord(ch) - ord("a") + 10
        if "A" <= ch <= "F": return ord(ch) - ord("A") + 10
        raise self._err("invalid hex digit")

    def _read_escape(self) -> str:
        c = self._peek()
        if c is None:
            raise self._err("unterminated escape")
        if c in "'\"\\": self._bump(1); return c
        if c == "n": self._bump(1); return "\n"
        if c == "r": self._bump(1); return "\r"
        if c == "t": self._bump(1); return "\t"
        if c == "x":
            self._bump(1)
            h1 = self._peek(); self._bump(1)
            h2 = self._peek(); self._bump(1)
            return chr(self._hexval(h1)*16 + self._hexval(h2))
        if c == "u":
            self._bump(1)
            val = 0
            for _ in range(4):
                h = self._peek(); self._bump(1)
                val = (val << 4) + self._hexval(h)
            return chr(val)
        # fallback: literal next char
        self._bump(1)
        return c

    def _literal(self) -> Literal:
        self._skip_ws()
        q = self._peek()
        if q not in ("'", '"'):
            raise self._err("expected quote")
        self._bump(1)
        out = []
        while not self._eof():
            c = self._peek()
            if c == q:
                self._bump(1)
                break
            if c == "\\":
                self._bump(1)
                out.append(self._read_escape())
            else:
                out.append(c)
                self._bump(1)
        else:
            raise self._err("unterminated string")
        return Literal("".join(out))

    def _class(self) -> CharClass:
        self._eat("[")
        neg = False
        if self._try_eat("^"):
            neg = True
        ranges: List[Tuple[int, int]] = []
        singles: List[str] = []

        def _read_char_in_class() -> str:
            if self._eof():
                raise self._err("unterminated char class")
            c = self._peek()
            if c == "\\":
                self._bump(1)
                return self._read_escape()
            if c == "]":
                raise self._err("unexpected ']' in char class")
            self._bump(1)
            return c

        while True:
            self._skip_ws()
            if self._try_eat("]"):
                break
            a = _read_char_in_class()
            if self._try_eat("-"):
                # range
                b = _read_char_in_class()
                if ord(a) > ord(b):
                    a, b = b, a
                ranges.append((ord(a), ord(b)))
            else:
                singles.append(a)

        return CharClass(negated=neg, ranges=ranges, singles=singles)

    # --- recursive descent for expressions ---

    def parse_grammar(self) -> PegGrammar:
        rules: Dict[str, RuleDef] = {}
        # allow zero or more rules; at least one required
        start_name: Optional[str] = None
        while True:
            self._skip_ws()
            if self._eof():
                break
            name = self._ident()
            self._eat("<-")
            expr = self._parse_expr()
            if name in rules:
                raise self._err(f"duplicate rule '{name}'")
            rules[name] = RuleDef(name, expr)
            if start_name is None:
                start_name = name
        if not rules:
            raise self._err("empty PEG grammar")
        return PegGrammar(rules=rules, start=start_name)  # type: ignore[arg-type]

    def _parse_expr(self) -> Node:
        seq = self._parse_seq()
        alts = [seq]
        while True:
            self._skip_ws()
            if not self._try_eat("/"):
                break
            alts.append(self._parse_seq())
        if len(alts) == 1:
            return alts[0]
        return Choice(alts)

    def _parse_seq(self) -> Node:
        items: List[Node] = []
        while True:
            self._skip_ws()
            # stop at ) or / or rule delimiter (EoF or next IDENT<-)
            ch = self._peek()
            if ch is None or ch in ")/":
                break
            # also stop if upcoming looks like IDENT "<-"
            if self._is_ident_start(ch):
                # try lookahead for "<-"
                save = self.i
                ident = self._ident()
                self._skip_ws()
                if self._starts("<-"):
                    # it's a rule head; rewind and stop sequence
                    self.i = save
                    break
                # not a rule head → treat as primary (Ref)
                self.i = save
            try:
                items.append(self._parse_prefix())
            except SyntaxError:
                break
        if not items:
            return Seq([])  # empty sequence (epsilon)
        if len(items) == 1:
            return items[0]
        return Seq(items)

    def _parse_prefix(self) -> Node:
        self._skip_ws()
        if self._try_eat("&"):
            return And(self._parse_suffix())
        if self._try_eat("!"):
            return Not(self._parse_suffix())
        return self._parse_suffix()

    def _parse_suffix(self) -> Node:
        node = self._parse_primary()
        self._skip_ws()
        if self._try_eat("?"):
            return Repeat(node, "?")
        if self._try_eat("*"):
            return Repeat(node, "*")
        if self._try_eat("+"):
            return Repeat(node, "+")
        return node

    def _parse_primary(self) -> Node:
        self._skip_ws()
        ch = self._peek()
        if ch == "(":
            self._bump(1)
            e = self._parse_expr()
            self._eat(")")
            return e
        if ch == ".":
            self._bump(1)
            return Any()
        if ch in ("'", '"'):
            return self._literal()
        if ch == "[":
            return self._class()
        # IDENT (reference)
        name = self._ident()
        return Ref(name)


def parse_peg_grammar(src: str) -> PegGrammar:
    """Parse a `%peg { ... }` block (without the surrounding braces)."""
    ts = _TS(src)
    return ts.parse_grammar()
