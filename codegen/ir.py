"""
lepta 코드 생성용 IR
=======

이 모듈은 파서 테이블(Tables)과 심볼 테이블(SymbolTable)을 받아,
타깃 언어(Rust/C) 방출기가 소비하기 쉬운 **중간표현(IR)** 로 변환한다.

설계 포인트
-----------
- 파서 테이블은 상태별 ACTION/GOTO를 **행(row) 단위**로 정리해 담는다.
- ACTION은 (term_id, kind, arg) 튜플의 리스트로 저장한다.
  * kind: 1=SHIFT, 2=REDUCE, 3=ACCEPT
  * arg : shift→next_state, reduce→prod_idx, accept→0
- GOTO는 (nonterm_id, next_state) 튜플의 리스트로 저장한다.
- 심볼 이름(디버그/문서용)도 포함한다.

주의
----
- prod 배열(좌변/우변 길이)은 **증강 생산 0번**을 포함한다.
- terms[0]은 항상 '$'(EOF)여야 한다(SymbolTable 규약).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from ..lalr.table import Tables
from ..lalr.symbols import SymbolTable
from ..grammar.ast import Grammar  # ← (옵션) 전역 %recover/%sync 읽기용

SHIFT, REDUCE, ACCEPT = 1, 2, 3

@dataclass
class PegTermIR:
    """PEG 토큰 1개에 대한 IR 메타 정보 (코드 생성용)."""
    term_id: int          # 단말 ID
    term_name: str        # 단말 이름(디버그/문서용)
    block_name: str       # %peg 블록 이름
    rule_name: str        # %peg 룰 이름
    trigger: Optional[str] = None  # 선택: 단일 문자 트리거

@dataclass 
class CodegenIR:
    """
    CodegenIR
    =========
    emit_*.py에서 사용하는 IR.

    Fields
    ------
    n_states      : 전체 상태 수
    n_terms       : 단말 개수(EOF 포함)
    n_nonterms    : 비단말 개수
    start_state   : 시작 상태 (증강 문법의 I0)
    terms         : 단말 이름 리스트 (id 순)
    nonterms      : 비단말 이름 리스트 (id 순: term_count..)
    prod_lhs      : 각 프로덕션의 LHS 심볼 ID (증강 포함)
    prod_rhs_len  : 각 프로덕션의 RHS 길이 (증강 포함)
    action_rows   : 상태별 ACTION 행. 각 행은 (term_id, kind, arg) 리스트
    goto_rows     : 상태별 GOTO 행. 각 행은 (nonterm_id, next_state) 리스트

    # (신규) 오류 복구 메타
    recover_mode  : None|"off"|"panic"
    sync_term_ids : 전역 %sync 라벨을 term id(u16)로 해석한 결과
    max_errors    : 연속 허용 오류 개수(런타임에서 활용 가능, 기본 5)
    """
    n_states: int
    n_terms: int
    n_nonterms: int
    start_state: int
    terms: List[str]
    nonterms: List[str]
    prod_lhs: List[int]
    prod_rhs_len: List[int]
    action_rows: List[List[Tuple[int, int, int]]]
    goto_rows: List[List[Tuple[int, int]]]

    # --- 오류 복구 ---
    recover_mode: Optional[str] = None
    sync_term_ids: List[int] = field(default_factory=list)
    max_errors: int = 5

    # --- PEG 메타데이터 for codegen ---
    peg_blocks: List[Tuple[str, str]] = field(default_factory=list)  # (block_name, src)
    peg_terms: List[PegTermIR] = field(default_factory=list)


def build_ir(sym: SymbolTable, tbl: Tables, g: Optional[Grammar] = None) -> CodegenIR:
    """
    build_ir(sym, tbl[, g]) -> CodegenIR
    ------------------------------------
    SymbolTable과 Tables를 **타깃 불문(c/rs) 공통 IR**로 변환한다.
    (신규) g가 주어지면 전역 %recover/%sync를 IR 메타로 담는다.

    정렬 규칙
    --------
    - ACTION/GOTO는 각 행(상태) 안에서 심볼 ID 기준 **오름차순**으로 정렬한다.
    - 단말/비단말 이름 배열은 ID 순서를 그대로 따른다.
    """
    # 1) 심볼 이름 배열 (id → name)
    terms = [sym.name_of(i) for i in range(sym.term_count)]
    nonterms = [sym.name_of(sym.term_count + j) for j in range(sym.nonterm_count)]

    # 2) ACTION/GOTO를 행 단위로 모으기
    action_rows: List[List[Tuple[int, int, int]]] = []
    goto_rows: List[List[Tuple[int, int]]] = []
    for s in range(tbl.n_states):
        # ACTION
        row_acts = []
        for (st, tid), (k, v) in tbl.action.items():
            if st != s:
                continue
            if k == 's':
                row_acts.append((tid, SHIFT, int(v)))
            elif k == 'r':
                row_acts.append((tid, REDUCE, int(v)))
            elif k == 'acc':
                row_acts.append((tid, ACCEPT, 0))
            else:
                # 예상치 못한 kind는 무시(또는 raise)
                continue
        row_acts.sort(key=lambda t: t[0])
        action_rows.append(row_acts)

        # GOTO
        row_g = []
        for (st, aid), next_st in tbl.goto.items():
            if st != s:
                continue
            row_g.append((aid, int(next_st)))
        row_g.sort(key=lambda t: t[0])
        goto_rows.append(row_g)

    # 3) (신규) 전역 %recover / %sync → IR 메타 채우기
    recover_mode: Optional[str] = None
    sync_term_ids: List[int] = []
    if g is not None:
        recover_mode = g.recover_mode or None

        # "EOF" 또는 "$"는 0으로 매핑
        def _to_tid(label: str) -> int:
            if label in ("EOF", "$"):
                return 0
            try:
                return terms.index(label)
            except ValueError:
                raise ValueError(f"IR build: unknown %sync label '{label}' (not a terminal): "
                                 f"known terms={terms}")

        for lab in (g.sync_labels or []):
            sync_term_ids.append(_to_tid(lab))

        # 중복 제거(보수적으로 순서 보존)
        if sync_term_ids:
            seen = set()
            uniq = []
            for t in sync_term_ids:
                if t not in seen:
                    uniq.append(t); seen.add(t)
            sync_term_ids = uniq

    peg_blocks: List[Tuple[str, str]] = []
    peg_terms: List[PegTermIR] = []
    if g is not None:
        # block_name -> src
        block_src: dict[str, str] = {pb.name: pb.src for pb in getattr(g, "decl_peg_blocks", [])}
        # term name -> id
        name_to_tid = {name: i for i, name in enumerate(terms)}
        # collect
        for bname, src in block_src.items():
            peg_blocks.append((bname, src))
        for pt in getattr(g, "decl_peg_tokens", []):
            term_name = pt.name
            try:
                tid = name_to_tid[term_name]
            except KeyError:
                raise ValueError(f"PEG token '{term_name}' is not in terminal set: terms={terms}")
            peg_terms.append(
                PegTermIR(
                    term_id=tid,
                    term_name=term_name,
                    block_name=pt.peg_ref[0],
                    rule_name=pt.peg_ref[1],
                    trigger=pt.trigger,
                )
            )

    return CodegenIR(
        n_states=tbl.n_states,
        n_terms=sym.term_count,
        n_nonterms=sym.nonterm_count,
        start_state=tbl.start_state,
        terms=terms,
        nonterms=nonterms,
        prod_lhs=list(tbl.prod_lhs_ids),
        prod_rhs_len=list(tbl.prod_rhs_len),
        action_rows=action_rows,
        goto_rows=goto_rows,

        # 오류 복구
        recover_mode=recover_mode,
        sync_term_ids=sync_term_ids,
        max_errors=5,

        # PEG meta for codegen
        peg_blocks=peg_blocks,
        peg_terms=peg_terms,
    )
