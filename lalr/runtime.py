# lepta/lalr/runtime.py
"""SLR(1) 파서 런타임(스택 머신).

- `Tables`(ACTION/GOTO)와 `SymbolTable`, 그리고 `Lexer`를 받아
  입력 문자열을 **accept/reject** 판정합니다.
- 에러 시, 해당 상태에서 가능한 단말(expected set)을 제시하는
  친절한 `SyntaxError` 메시지를 던집니다.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, Dict
from dataclasses import dataclass

from .table import Tables
from .symbols import SymbolTable
from ..lex import Lexer, LexTok

def _line_bounds(src: str, pos: int) -> Tuple[int, int]:
    """pos가 속한 라인의 [start, end) 범위를 반환."""
    start = src.rfind("\n", 0, pos)
    start = 0 if start < 0 else start + 1
    end = src.find("\n", pos)
    end = len(src) if end < 0 else end
    return start, end

def _caret_snippet(src: str, pos: int) -> str:
    """해당 절대 오프셋 pos에 캐럿(^)을 찍은 스니펫을 생성."""
    start, end = _line_bounds(src, pos)
    line = src[start:end]
    col = (pos - start) + 1
    caret = " " * (col - 1) + "^"
    return f"{line}\n{caret}"

def parse_string(text: str, tables: Tables, sym: SymbolTable, lexer: Lexer) -> bool:
    """SLR(1) 테이블과 렉서를 사용하여 입력 문자열을 파싱합니다.

    Parameters
    ----------
    text : str
        파싱할 원문.
    tables : Tables
        SLR(1) ACTION/GOTO 테이블 묶음.
    sym : SymbolTable
        심볼 이름 ↔ ID 조회용 테이블.
    lexer : Lexer
        현재 문법(AST)로 구성된 렉서 인스턴스.

    Returns
    -------
    bool
        파싱 성공 시 True. 실패 시 `SyntaxError`를 발생시킵니다.

    Notes
    -----
    - 값(semantic value) 스택은 아직 사용하지 않습니다(MVP).
    - 에러 시 메시지에는 **다음 토큰 위치**와 함께, 해당 상태에서 가능한
      **expected 단말 집합**을 포함합니다.
    """
    # 내부 상태 스택(정수): 시작 상태에서 시작
    state_stack: List[int] = [tables.start_state]

    # 입력 토큰 스트림 준비
    lexer.reset(text)
    look: Optional[LexTok] = lexer.peek()

    def ensure_term_id(tok: Optional[LexTok]) -> int:
        if tok is None:
            return sym.eof_id
        try:
            return sym.id_of(tok.type)
        except KeyError:
            # 심볼테이블에 없는 단말(예: 렉서/문법 불일치)
            raise SyntaxError(
                f"Unknown token kind {tok.type!r} at {tok.line}:{tok.col}\n"
                + _caret_snippet(text, tok.pos)
            )

    while True:
        s = state_stack[-1]
        a_id = ensure_term_id(look)
        act = tables.action.get((s, a_id))

        if act is None:
            # 에러: expected set 수집
            expected: List[str] = []
            for (st, tid), op in tables.action.items():
                if st == s:
                    try:
                        expected.append(sym.name_of(tid))
                    except Exception:
                        expected.append(f"#{tid}")
            expected_sorted = ", ".join(sorted(set(expected)))
            if look is None:
                # EOF에서 실패
                pos = len(text)
                line = text.count("\n") + 1
                col = (pos - _line_bounds(text, pos)[0]) + 1
                raise SyntaxError(
                    "Parse error at EOF: expected one of "
                    f"{{{expected_sorted}}}\n" + _caret_snippet(text, pos)
                )
            else:
                raise SyntaxError(
                    f"Parse error at {look.line}:{look.col}: unexpected {look.type!r}, "
                    f"expected one of {{{expected_sorted}}}\n" + _caret_snippet(text, look.pos)
                )

        kind, arg = act

        if kind == 's':  # shift
            # 상태 push + 토큰 소비
            state_stack.append(arg)
            lexer.next()  # consume
            look = lexer.peek()
            continue

        if kind == 'r':  # reduce by production #arg
            p_idx = arg
            rhs_len = tables.prod_rhs_len[p_idx]
            lhs_id = tables.prod_lhs_ids[p_idx]
            # 상태 pop
            for _ in range(rhs_len):
                if len(state_stack) <= 1:
                    # 방어적: underflow 방지
                    break
                state_stack.pop()
            t = state_stack[-1]
            goto_state = tables.goto.get((t, lhs_id))
            if goto_state is None:
                # 구조적 오류(테이블 일관성 문제)
                lhs_name = sym.name_of(lhs_id)
                raise RuntimeError(f"GOTO missing for state={t}, lhs={lhs_name}")
            state_stack.append(goto_state)
            continue

        if kind == 'acc':
            return True

        # 방어적
        raise RuntimeError(f"Unknown ACTION kind: {kind}")

