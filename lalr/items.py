# lepta/lalr/items.py
"""LR(0) DFA와 SLR(1)/LALR(1) 테이블 작성 + S/R 충돌을 precedence로 해소.

이 모듈은 EBNF→BNF로 전개된 문법(BNF)을 입력으로 받아
- LR(0) 아이템/클로저/고토
- 상태 DFA 구성
- SLR(1) ACTION/GOTO 테이블 산출
- LALR(1) ACTION/GOTO 테이블 산출
- 발생한 shift/reduce 충돌을 precedence/associativity 규칙으로 자동 해소
을 수행한다.

주의:
- 증강 시작기호 S'는 심볼테이블에 존재하지 않으므로, 메타데이터(ID 변환) 시
  원래 시작기호의 ID로 **대체 매핑**한다(accept 판단에는 영향 없음).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Iterable, Optional

from ..grammar.transform import BNF, Production
from .symbols import SymbolTable
from .table import Tables
from .first_follow import compute_nullable_first_follow

Assoc = str  # 'left' | 'right' | 'nonassoc'

# ---------- 공통 유틸 -------------
def _is_term(name: str, bnf: BNF) -> bool:
    return name in bnf.terms

def _is_nonterm(name: str, bnf: BNF) -> bool:
    return name in bnf.nonterms

# ---------- precedence 해석 유틸 -------------

def _build_prec_tables(
    bnf: BNF,
    sym: SymbolTable
) -> Tuple[Dict[int, Tuple[int, Assoc]], Dict[str, Tuple[int, Assoc]]]:
    """
    precedence 선언으로부터 두 테이블을 만듭니다.
    - term_prec[term_id] -> (level, assoc)
    - label_prec[label_str] -> (level, assoc)   # 가상 라벨/리터럴 포함
    선언의 **아래쪽일수록 레벨이 높다**(더 강함).
    """
    term_prec: Dict[int, Tuple[int, Assoc]] = {}
    label_prec: Dict[str, Tuple[int, Assoc]] = {}

    level = 0
    for decl in bnf.precedences:
        level += 1
        assoc = decl.assoc
        for lab in decl.labels:
            label_prec[lab] = (level, assoc)
            # lab 이 실제 단말이면 term_id 에도 매핑
            try:
                tid = sym.id_of(lab)
                if sym.is_term_id(tid):
                    term_prec[tid] = (level, assoc)
            except KeyError:
                # 가상 라벨 / 아직 심볼테이블에 없는 리터럴: label_prec 로만 유지
                pass
    return term_prec, label_prec

def _rightmost_term_id(rhs: List[str], sym: SymbolTable) -> Optional[int]:
    """rhs에서 가장 오른쪽 단말의 term_id 를 찾습니다. 없으면 None."""
    for name in reversed(rhs):
        try:
            sid = sym.id_of(name)
        except KeyError:
            continue
        if sym.is_term_id(sid):
            return sid
    return None

def _prod_prec_key(
    p: Production,
    sym: SymbolTable,
    term_prec: Dict[int, Tuple[int, Assoc]],
    label_prec: Dict[str, Tuple[int, Assoc]]
) -> Optional[Tuple[int, Assoc]]:
    """
    프로덕션 p의 우선순위 키를 계산합니다.
    - p.prec_label 이 있으면 label_prec 에서 찾음
    - 없으면 오른쪽 끝 단말의 term_prec 사용
    - 둘 다 없으면 None
    """
    if p.prec_label is not None:
        return label_prec.get(p.prec_label)
    tid = _rightmost_term_id(p.rhs, sym)
    if tid is None:
        return None
    return term_prec.get(tid)

def resolve_sr_with_precedence(
    action: Dict[Tuple[int,int], Tuple[str,int]],
    conflicts: List[Tuple[int, int, Tuple[str, str]]],
    prods: List[Production],
    sym: SymbolTable,
    bnf: BNF,
    *,
    debug: bool = False,
    # 추가 힌트(선택): SLR/LALR에서 reduce 프로덕션을 특정하기 위한 상태 정보
    slr_state_items: Optional[List[Set["Item"]]] = None,
    slr_follow: Optional[Dict[str, Set[str]]] = None,
    lalr_states: Optional[List[Dict[Tuple[int,int], Set[str]]]] = None,
) -> None:
    """
    ACTION 테이블에서 발생한 shift/reduce 충돌을 우선순위/결합성으로 해소합니다.
    - action[(state, term_id)] = ('s', next) | ('r', p) | ('acc', 0)
    - conflicts: (state, term_id, ('shift','reduce')) 등 기록(여기서 일부 제거/치환)
    - slr_state_items/follow 또는 lalr_states가 주어지면 reduce 프로덕션을 확정할 수 있습니다.
    """
    term_prec, label_prec = _build_prec_tables(bnf, sym)
    to_remove: List[int] = []

    def _find_reduce_prod_slr(st: int, term_id: int) -> Optional[int]:
        if slr_state_items is None or slr_follow is None:
            return None
        I = slr_state_items[st]
        term_name = sym.name_of(term_id)
        candidates: Set[int] = set()
        for it in I:
            p = prods[it.prod_idx]
            if it.dot == len(p.rhs):
                # FOLLOW(A)에 term_name 이 있으면 reduce 후보
                if term_name in slr_follow.get(p.lhs, set()):
                    candidates.add(it.prod_idx)
        if len(candidates) == 1:
            return next(iter(candidates))
        return None  # 모호(R/R) 등은 여기서 다루지 않음

    def _find_reduce_prod_lalr(st: int, term_id: int) -> Optional[int]:
        if lalr_states is None:
            return None
        state = lalr_states[st]
        term_name = sym.name_of(term_id)
        candidates: Set[int] = set()
        for (pi, d), la in state.items():
            p = prods[pi]
            if d == len(p.rhs) and term_name in la:
                candidates.add(pi)
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    for idx, (st, term_id, kinds) in enumerate(list(conflicts)):
        if kinds not in (('shift','reduce'), ('reduce','shift')):
            continue

        # reduce 프로덕션 인덱스 확정
        reduce_p: Optional[int] = None

        # 1) action 칸 자체가 reduce로 채워져 있다면 바로 사용
        a = action.get((st, term_id))
        if a is not None and a[0] == 'r':
            reduce_p = a[1]

        # 2) 상태/LA를 이용해 유도 (SLR/LALR 전용)
        if reduce_p is None:
            reduce_p = _find_reduce_prod_lalr(st, term_id) or _find_reduce_prod_slr(st, term_id)

        if reduce_p is None:
            # reduce 후보를 특정할 수 없으면 정책 유지(shift)하며 충돌 표시는 남겨 두지 않고 제거
            # (사용자 입장에선 precedence 대상 아님)
            to_remove.append(idx)
            if debug:
                try:
                    tn = sym.name_of(term_id)
                except Exception:
                    tn = f"term#{term_id}"
                print(f"[prec] state {st}, on {tn}: cannot identify reduce prod → keep SHIFT")
            continue

        prod = prods[reduce_p]
        prod_key = _prod_prec_key(prod, sym, term_prec, label_prec)
        term_key = term_prec.get(term_id)

        if not prod_key or not term_key:
            # prec 정보가 부족 → 기존 정책 유지(shift 우선)
            to_remove.append(idx)
            if debug:
                tn = sym.name_of(term_id)
                print(f"[prec] state {st}, on {tn}: no precedence info → keep SHIFT")
            continue

        (pl, pa), (tl, ta) = prod_key, term_key
        tn = sym.name_of(term_id)

        if tl > pl:
            # lookahead 단말이 더 강함 → SHIFT
            # (대개 이미 SHIFT 이므로 그대로 두고 충돌만 제거)
            to_remove.append(idx)
            if debug:
                print(f"[prec] state {st}, on {tn}: prefer SHIFT (term level {tl} > prod {pl})")
        elif tl < pl:
            # prod 쪽이 더 강함 → REDUCE
            action[(st, term_id)] = ('r', reduce_p)
            to_remove.append(idx)
            if debug:
                print(f"[prec] state {st}, on {tn}: prefer REDUCE (prod level {pl} > term {tl})")
        else:
            # 같은 레벨 → assoc으로 판정
            if ta == 'left':
                action[(st, term_id)] = ('r', reduce_p)
                to_remove.append(idx)
                if debug:
                    print(f"[prec] state {st}, on {tn}: same level, LEFT → REDUCE")
            elif ta == 'right':
                # shift 유지
                to_remove.append(idx)
                if debug:
                    print(f"[prec] state {st}, on {tn}: same level, RIGHT → SHIFT")
            else:
                # nonassoc → 에러 칸(아예 비움)
                if (st, term_id) in action:
                    del action[(st, term_id)]
                to_remove.append(idx)
                if debug:
                    print(f"[prec] state {st}, on {tn}: same level, NONASSOC → ERROR")

    # 해소된 충돌들 제거
    for i in sorted(set(to_remove), reverse=True):
        try:
            conflicts.pop(i)
        except IndexError:
            pass


# ---------- LR(0) 아이템 ----------
@dataclass(frozen=True)
class Item:
    """LR(0) 아이템: [A -> α · β]"""
    prod_idx: int
    dot: int
    def __str__(self) -> str:
        return f"(p={self.prod_idx}, dot={self.dot})"

def _kernel_key_lr0(items: Set[Item]) -> Tuple[Tuple[int, int], ...]:
    """LR(0) 커널 아이템만 정렬해 커널 키 생성."""
    ker = [it for it in items if it.dot != 0]
    if not ker:
        ker = list(items)  # 초기 상태 보정
    ker.sort(key=lambda x: (x.prod_idx, x.dot))
    return tuple((it.prod_idx, it.dot) for it in ker)

def _closure_lr0(items: Set[Item],
                 prods: List[Production],
                 nonterms: Set[str],
                 lhs_to_prods: Dict[str, List[int]]) -> Set[Item]:
    I = set(items)
    changed = True
    while changed:
        changed = False
        for it in list(I):
            p = prods[it.prod_idx]
            if it.dot < len(p.rhs):
                symb = p.rhs[it.dot]
                if symb in nonterms:
                    for pj in lhs_to_prods.get(symb, []):
                        new_item = Item(pj, 0)
                        if new_item not in I:
                            I.add(new_item)
                            changed = True
    return I

def _goto_lr0(items: Set[Item], X: str, prods: List[Production]) -> Set[Item]:
    J = set()
    for it in items:
        p = prods[it.prod_idx]
        if it.dot < len(p.rhs) and p.rhs[it.dot] == X:
            J.add(Item(it.prod_idx, it.dot + 1))
    return J


# ---------- SLR(1) 테이블 빌더 ----------

def build_slr_tables(bnf: BNF, sym: SymbolTable, follow: Dict[str, Set[str]]) -> Tables:
    """
    BNF + SymbolTable + FOLLOW 집합으로 **SLR(1) ACTION/GOTO** 테이블을 생성한다.
    절차: 증강문법 추가 → LR(0) DFA → FOLLOW 기반 reduce → 테이블 생성
    """
    # --- 0) 증강문법 ---
    start = bnf.start
    aug_start = "__S'"
    prods: List[Production] = [Production(aug_start, [start])] + list(bnf.prods)
    nonterms = set(bnf.nonterms) | {aug_start}

    # LHS -> 프로덕션 인덱스
    lhs_to_prods: Dict[str, List[int]] = {}
    for idx, p in enumerate(prods):
        lhs_to_prods.setdefault(p.lhs, []).append(idx)

    # --- 1) I0 ---
    I0 = _closure_lr0({Item(0, 0)}, prods, nonterms, lhs_to_prods)
    states: List[Set[Item]] = []
    state_index: Dict[Tuple[Tuple[int, int], ...], int] = {}

    def add_state(items: Set[Item]) -> int:
        ker = _kernel_key_lr0(items)
        if ker in state_index:
            return state_index[ker]
        idx = len(states)
        states.append(items)
        state_index[ker] = idx
        return idx

    start_state = add_state(I0)

    # --- 2) DFA ---
    worklist: List[int] = [start_state]
    transitions: Dict[Tuple[int, str], int] = {}

    while worklist:
        s = worklist.pop()
        I = states[s]
        next_syms: Set[str] = set()
        for it in I:
            p = prods[it.prod_idx]
            if it.dot < len(p.rhs):
                next_syms.add(p.rhs[it.prod_idx - it.prod_idx + it.dot])  # == p.rhs[it.dot]
        for X in sorted(next_syms):
            J_core = _goto_lr0(I, X, prods)
            if not J_core:
                continue
            J = _closure_lr0(J_core, prods, nonterms, lhs_to_prods)
            j_idx = add_state(J)
            if (s, X) not in transitions:
                transitions[(s, X)] = j_idx
                if j_idx == len(states) - 1:
                    worklist.append(j_idx)

    # --- 3) ACTION/GOTO ---
    action: Dict[Tuple[int, int], Tuple[str, int]] = {}
    goto: Dict[Tuple[int, int], int] = {}
    conflicts: List[Tuple[int, int, Tuple[str, str]]] = []

    def sid(name: str) -> int:
        if name == aug_start:
            return sym.start_id
        return sym.id_of(name)

    prod_lhs_ids: List[int] = [sid(p.lhs) for p in prods]
    prod_rhs_len: List[int] = [len(p.rhs) for p in prods]

    for s, I in enumerate(states):
        # shift/goto
        next_syms: Set[str] = set()
        for it in I:
            p = prods[it.prod_idx]
            if it.dot < len(p.rhs):
                next_syms.add(p.rhs[it.dot])

        for X in next_syms:
            j = transitions.get((s, X))
            if j is None:
                continue
            xid = sid(X)
            if sym.is_term_id(xid):
                key = (s, xid)
                prev = action.get(key)
                if prev is not None and prev != ('s', j):
                    if prev[0] == 'r':
                        conflicts.append((s, xid, ('shift', 'reduce')))
                    else:
                        conflicts.append((s, xid, ('shift', prev[0])))
                action[key] = ('s', j)
            else:
                goto[(s, xid)] = j

        # reduce/accept
        for it in I:
            p = prods[it.prod_idx]
            if it.dot == len(p.rhs):
                A = p.lhs
                if A == aug_start:
                    key = (s, sym.eof_id)
                    prev = action.get(key)
                    if prev and prev != ('acc', 0):
                        conflicts.append((s, sym.eof_id, ('accept', prev[0])))
                    action[key] = ('acc', 0)
                else:
                    for a in follow[A]:
                        aid = sid(a)
                        key = (s, aid)
                        prev = action.get(key)
                        if prev is None:
                            action[key] = ('r', it.prod_idx)
                        else:
                            if prev[0] == 's':
                                conflicts.append((s, aid, ('shift', 'reduce')))
                            elif prev[0] == 'r' and prev[1] != it.prod_idx:
                                conflicts.append((s, aid, ('reduce', 'reduce')))
                                if it.prod_idx < prev[1]:
                                    action[key] = ('r', it.prod_idx)

    # precedence로 S/R 해소
    resolve_sr_with_precedence(
        action, conflicts, prods, sym, bnf,
        debug=False,
        slr_state_items=states,
        slr_follow=follow,
    )

    # --- 4) 디버그 문자열 ---
    def fmt_item(it: Item) -> str:
        p = prods[it.prod_idx]
        rhs = list(p.rhs)
        rhs.insert(it.dot, "·")
        return f"[{p.lhs} -> {' '.join(rhs)}]"

    state_items: List[List[str]] = []
    for I in states:
        state_items.append([fmt_item(it) for it in sorted(I, key=lambda x: (x.prod_idx, x.dot))])

    return Tables(
        action=action,
        goto=goto,
        start_state=start_state,
        n_states=len(states),
        prod_lhs_ids=prod_lhs_ids,
        prod_rhs_len=prod_rhs_len,
        state_items=state_items,
        conflicts=conflicts,
    )


# ---------- LALR(1) 빌더 ----------

@dataclass(frozen=True)
class LR1Item:
    """LR(1) 아이템: [A -> α · β, lookaheads]"""
    prod_idx: int
    dot: int
    look: frozenset[str]

def _state_repr_lr1(items: Dict[Tuple[int, int], Set[str]]) -> Tuple[Tuple[int, int, Tuple[str, ...]], ...]:
    """상태(아이템 맵)를 해시 가능 튜플로 직렬화 (canonical 상태 비교용)."""
    rows = []
    for (pi, d), la in items.items():
        rows.append((pi, d, tuple(sorted(la))))
    rows.sort()
    return tuple(rows)

def _kernel_core_key_lr1(items: Dict[Tuple[int, int], Set[str]], prods: List[Production]) -> Tuple[Tuple[int, int], ...]:
    """same-core 병합용 커널 코어 키(lookahead 제외)."""
    core = []
    for (pi, d) in items.keys():
        # 커널: dot>0 또는 증강 시작 아이템(p0,0)
        if d != 0 or pi == 0:
            core.append((pi, d))
    core.sort()
    return tuple(core)

def _first_seq(tail: Iterable[str],
               la: Set[str],
               bnf: BNF,
               first: Dict[str, Set[str]],
               nullable: Set[str]) -> Set[str]:
    """FIRST(β a) 계산. β가 전부 nullable이면 la(lookahead)도 포함."""
    res: Set[str] = set()
    all_nullable = True
    for X in tail:
        if _is_term(X, bnf):
            res.add(X)
            all_nullable = False
            break
        # nonterm
        res |= first.get(X, set())
        if X not in nullable:
            all_nullable = False
            break
    if all_nullable:
        res |= la
    return res

def _closure_lr1(state: Dict[Tuple[int, int], Set[str]],
                 prods: List[Production],
                 bnf: BNF,
                 lhs_to_prods: Dict[str, List[int]],
                 first: Dict[str, Set[str]],
                 nullable: Set[str]) -> None:
    """in-place 고정점: LR(1) closure. state[(pi,d)] = lookahead set"""
    changed = True
    while changed:
        changed = False
        # snapshot of current items to allow in-loop expansion
        for (pi, d), look in list(state.items()):
            p = prods[pi]
            if d < len(p.rhs):
                B = p.rhs[d]
                if _is_nonterm(B, bnf):
                    beta = p.rhs[d+1:]
                    LA = _first_seq(beta, look, bnf, first, nullable)
                    for pj in lhs_to_prods.get(B, []):
                        key = (pj, 0)
                        s = state.setdefault(key, set())
                        old_len = len(s)
                        s |= LA
                        if len(s) != old_len:
                            changed = True

def _goto_lr1(state: Dict[Tuple[int, int], Set[str]],
              X: str,
              prods: List[Production],
              bnf: BNF,
              lhs_to_prods: Dict[str, List[int]],
              first: Dict[str, Set[str]],
              nullable: Set[str]) -> Dict[Tuple[int, int], Set[str]]:
    """LR(1) goto: 점 뒤가 X인 아이템 전진 + closure."""
    nxt: Dict[Tuple[int, int], Set[str]] = {}
    for (pi, d), look in state.items():
        p = prods[pi]
        if d < len(p.rhs) and p.rhs[d] == X:
            key = (pi, d+1)
            s = nxt.setdefault(key, set())
            s |= look
    if not nxt:
        return {}
    _closure_lr1(nxt, prods, bnf, lhs_to_prods, first, nullable)
    return nxt

def build_lalr_tables(bnf: BNF, sym: SymbolTable) -> Tables:
    """
    build_lalr_tables
    =================
    Canonical LR(1)을 만든 뒤, same kernel(core)을 **병합**하여 LALR(1) 테이블을 생성합니다.

    절차
    ----
    1) 증강문법 S' -> S, 초기 아이템 [S'->·S, {$}] closure
    2) canonical LR(1) DFA 생성 (상태 동일성 = 아이템+lookahead 동일)
    3) same-core 병합: 커널(core: dot>0 또는 p0,0)의 (prod_idx,dot) 집합이 같은 상태들을 하나로 병합
       - 병합 시 커널 아이템의 lookahead를 합집합
       - 병합 커널에서 한 번 더 closure (lookahead 전파 보정)
    4) 합쳐진 상태들로 ACTION/GOTO 작성
       - shift/goto: 전이 재배선
       - reduce: [A->α·, L]의 L의 각 토큰에 대해 reduce
       - accept: [S'->S·, L] and '$'∈L 일 때 EOF에서 accept
    5) 충돌 기록(shift 우선, r/r는 더 작은 규칙 우선)
    6) precedence/assoc으로 SR 충돌 자동 해소
    """
    # --- 사전 계산: FIRST/FOLLOW/NULLABLE ---
    ff = compute_nullable_first_follow(bnf, sym)
    first = ff.first     # Dict[nonterm, Set[term]]
    nullable = ff.nullable  # Set[nonterm]
    # follow = ff.follow  # (필요 없음: LALR는 lookahead 집합을 직접 사용)

    # --- 0) 증강문법/테이블 준비 ---
    start = bnf.start
    aug_start = "__S'"
    prods: List[Production] = [Production(aug_start, [start])] + list(bnf.prods)
    nonterms = set(bnf.nonterms) | {aug_start}
    eof_name = sym.name_of(sym.eof_id)

    # LHS -> 프로덕션 인덱스들
    lhs_to_prods: Dict[str, List[int]] = {}
    for idx, p in enumerate(prods):
        lhs_to_prods.setdefault(p.lhs, []).append(idx)

    # --- 1) 초기 상태 (canonical LR(1)) ---
    I0: Dict[Tuple[int, int], Set[str]] = {(0, 0): {eof_name}}
    _closure_lr1(I0, prods, bnf, lhs_to_prods, first, nullable)

    # 상태 저장: Dict[(pi,dot)->set(la)]
    states: List[Dict[Tuple[int, int], Set[str]]] = []
    state_index: Dict[Tuple[Tuple[int, int, Tuple[str, ...]], ...], int] = {}

    def add_state(st: Dict[Tuple[int, int], Set[str]]) -> int:
        rep = _state_repr_lr1(st)
        if rep in state_index:
            return state_index[rep]
        idx = len(states)
        states.append(st)
        state_index[rep] = idx
        return idx

    start_state = add_state(I0)

    # --- 2) canonical LR(1) DFA ---
    worklist: List[int] = [start_state]
    transitions: Dict[Tuple[int, str], int] = {}

    # 모든 심볼(단말+비단말) 후보 수집 함수
    def next_symbols(st: Dict[Tuple[int, int], Set[str]]) -> Set[str]:
        S: Set[str] = set()
        for (pi, d) in st.keys():
            p = prods[pi]
            if d < len(p.rhs):
                S.add(p.rhs[d])
        return S

    while worklist:
        s = worklist.pop()
        I = states[s]
        for X in sorted(next_symbols(I)):
            J = _goto_lr1(I, X, prods, bnf, lhs_to_prods, first, nullable)
            if not J:
                continue
            j_idx = add_state(J)
            if (s, X) not in transitions:
                transitions[(s, X)] = j_idx
                if j_idx == len(states) - 1:
                    worklist.append(j_idx)

    # --- 3) same-core(LALR) 병합 ---
    groups: Dict[Tuple[Tuple[int, int], ...], List[int]] = {}
    for i, st in enumerate(states):
        core = _kernel_core_key_lr1(st, prods)
        groups.setdefault(core, []).append(i)

    merged_states: List[Dict[Tuple[int, int], Set[str]]] = []
    group_to_newid: Dict[Tuple[Tuple[int, int], ...], int] = {}

    for core, members in groups.items():
        # 커널 아이템의 LA 합집합
        kernel_map: Dict[Tuple[int, int], Set[str]] = {k: set() for k in core}
        for idx in members:
            st = states[idx]
            for (pi, d) in core:
                la = st.get((pi, d), set())
                kernel_map[(pi, d)].update(la)
        # 커널에서 closure 수행 (비커널 포함해 최종 상태 도출)
        _closure_lr1(kernel_map, prods, bnf, lhs_to_prods, first, nullable)
        # 등록
        nid = len(merged_states)
        merged_states.append(kernel_map)
        group_to_newid[core] = nid

    # canonical state → merged state 매핑
    old_to_new: Dict[int, int] = {}
    for i, st in enumerate(states):
        core = _kernel_core_key_lr1(st, prods)
        old_to_new[i] = group_to_newid[core]

    # 전이 재배선
    merged_trans: Dict[Tuple[int, str], int] = {}
    for (i, X), j in transitions.items():
        mi, mj = old_to_new[i], old_to_new[j]
        merged_trans[(mi, X)] = mj

    # 시작 상태 재계산
    m_start_state = old_to_new[start_state]

    # --- 4) ACTION/GOTO 생성 ---
    action: Dict[Tuple[int, int], Tuple[str, int]] = {}
    goto: Dict[Tuple[int, int], int] = {}
    conflicts: List[Tuple[int, int, Tuple[str, str]]] = []

    def sid(name: str) -> int:
        if name == aug_start:
            return sym.start_id
        return sym.id_of(name)

    prod_lhs_ids: List[int] = [sid(p.lhs) for p in prods]
    prod_rhs_len: List[int] = [len(p.rhs) for p in prods]

    # shift/goto (전이 기반)
    for (s, X), t in merged_trans.items():
        xid = sid(X)
        if sym.is_term_id(xid):
            key = (s, xid)
            prev = action.get(key)
            if prev is not None and prev != ('s', t):
                if prev[0] == 'r':
                    conflicts.append((s, xid, ('shift', 'reduce')))
                else:
                    conflicts.append((s, xid, ('shift', prev[0])))
            action[key] = ('s', t)
        else:
            goto[(s, xid)] = t

    # reduce / accept (lookahead 사용)
    for s, st in enumerate(merged_states):
        for (pi, d), la_set in st.items():
            p = prods[pi]
            if d == len(p.rhs):
                if p.lhs == aug_start:
                    # accept: lookahead에 '$'가 있는 경우
                    eof_name = sym.name_of(sym.eof_id)
                    if eof_name in la_set:
                        key = (s, sym.eof_id)
                        prev = action.get(key)
                        if prev and prev != ('acc', 0):
                            conflicts.append((s, sym.eof_id, ('accept', prev[0])))
                        action[key] = ('acc', 0)
                else:
                    for a in la_set:
                        aid = sid(a)
                        key = (s, aid)
                        prev = action.get(key)
                        if prev is None:
                            action[key] = ('r', pi)
                        else:
                            if prev[0] == 's':
                                conflicts.append((s, aid, ('shift', 'reduce')))
                            elif prev[0] == 'r' and prev[1] != pi:
                                conflicts.append((s, aid, ('reduce', 'reduce')))
                                if pi < prev[1]:
                                    action[key] = ('r', pi)

    # precedence로 S/R 해소
    resolve_sr_with_precedence(
        action, conflicts, prods, sym, bnf,
        debug=False,
        lalr_states=merged_states,
    )

    # --- 5) 디버그 문자열 (lookahead 포함) ---
    def fmt_item_lr1(pi: int, d: int, las: Set[str]) -> str:
        p = prods[pi]
        rhs = list(p.rhs)
        rhs.insert(d, "·")
        las_s = ", ".join(sorted(las)) if las else "∅"
        return f"[{p.lhs} -> {' '.join(rhs)} , {{{las_s}}}]"

    state_items: List[List[str]] = []
    for st in merged_states:
        items_str = []
        for (pi, d) in sorted(st.keys(), key=lambda x: (x[0], x[1])):
            items_str.append(fmt_item_lr1(pi, d, st[(pi, d)]))
        state_items.append(items_str)

    return Tables(
        action=action,
        goto=goto,
        start_state=m_start_state,
        n_states=len(merged_states),
        prod_lhs_ids=prod_lhs_ids,
        prod_rhs_len=prod_rhs_len,
        state_items=state_items,
        conflicts=conflicts,
    )
