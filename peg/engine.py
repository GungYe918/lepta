# lepta/peg/engine.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from .ast import (
    Literal, CharClass, Any, Ref, And, Not, Repeat, Seq, Choice,
    RuleDef, PegGrammar, Node
)

# Packrat engine:
# - Memoize only rule applications (rule_name, pos) -> (ok, end_pos)
# - Left recursion is not supported (typical PEG restriction).
# - Evaluation functions are pure and do not build trees (tokenization mode).

def _class_match(cc: CharClass, ch: str) -> bool:
    ok = False
    cp = ord(ch)
    for (lo, hi) in cc.ranges:
        if lo <= cp <= hi:
            ok = True
            break
    if not ok and cc.singles:
        if ch in cc.singles:
            ok = True
    return (not ok) if cc.negated else ok


class Packrat:
    def __init__(self, g: PegGrammar):
        self.g = g
        # memo: (rule_name, pos) -> (visited_flag:int, ok:bool, end:int)
        # visited_flag: 0=not visited, 1=in progress, 2=done
        self.memo: Dict[Tuple[str, int], Tuple[int, bool, int]] = {}

    # ---- Public entrypoint for one rule ----
    def parse(self, rule_name: str, text: str, pos: int = 0) -> Tuple[bool, int]:
        # Clear memo per top-level run (callers usually cache across tries if needed)
        self.memo.clear()
        ok, end = self._apply_rule(rule_name, text, pos)
        return ok, end

    # ---- Rule application with memoization ----
    def _apply_rule(self, name: str, text: str, pos: int) -> Tuple[bool, int]:
        key = (name, pos)
        m = self.memo.get(key)
        if m is not None:
            flag, ok, end = m
            if flag == 1:
                # left recursion or re-entry → fail (PEG disallows left recursion)
                return False, pos
            return ok, end

        # mark in-progress
        self.memo[key] = (1, False, pos)
        rule = self.g.require_rule(name)
        ok, end = self._eval(rule.expr, text, pos)
        # store result
        self.memo[key] = (2, ok, end)
        return ok, end

    # ---- Evaluator for expressions ----
    def _eval(self, node: Node, text: str, pos: int) -> Tuple[bool, int]:
        if isinstance(node, Literal):
            if text.startswith(node.text, pos):
                return True, pos + len(node.text)
            return False, pos

        if isinstance(node, Any):
            if pos < len(text):
                # consume exactly one Unicode scalar (Python char)
                c = text[pos]
                # but Python string index is already char-based
                return True, pos + 1
            return False, pos

        if isinstance(node, CharClass):
            if pos < len(text):
                c = text[pos]
                if _class_match(node, c):
                    return True, pos + 1
            return False, pos

        if isinstance(node, Ref):
            return self._apply_rule(node.name, text, pos)

        if isinstance(node, And):
            ok, _ = self._eval(node.node, text, pos)
            return (ok, pos)

        if isinstance(node, Not):
            ok, _ = self._eval(node.node, text, pos)
            return ((not ok), pos)

        if isinstance(node, Repeat):
            if node.kind == "?":
                ok, end = self._eval(node.node, text, pos)
                return (True, end if ok else pos)
            elif node.kind == "*":
                cur = pos
                while True:
                    ok, end = self._eval(node.node, text, cur)
                    if not ok or end == cur:
                        break
                    cur = end
                return True, cur
            elif node.kind == "+":
                ok, end = self._eval(node.node, text, pos)
                if not ok:
                    return False, pos
                cur = end
                while True:
                    ok2, end2 = self._eval(node.node, text, cur)
                    if not ok2 or end2 == cur:
                        break
                    cur = end2
                return True, cur
            else:
                raise AssertionError(f"unknown repeat kind {node.kind!r}")

        if isinstance(node, Seq):
            cur = pos
            for it in node.items:
                ok, end = self._eval(it, text, cur)
                if not ok:
                    return False, pos
                cur = end
            return True, cur

        if isinstance(node, Choice):
            for it in node.alts:
                ok, end = self._eval(it, text, pos)
                if ok:
                    return True, end
            return False, pos

        raise AssertionError(f"unknown node: {node!r}")
