from __future__ import annotations
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass
from ..grammar.transform import BNF, Production
from .symbols import SymbolTable


@dataclass
class FFResult:
    """
    FFResult
    ========
    FIRST/FOLLOW/NULLABLE 계산 결과를 담는 단순 컨테이너입니다.

    - nullable: ε-생산 가능한 비단말 집합 (이름 기반)
    - first: 각 **심볼 이름** → FIRST 집합(단말 이름들의 집합)
      * 비단말 A: FIRST(A)
      * 단말 a: FIRST(a) = { a }
    - follow: 각 **비단말 이름** → FOLLOW 집합(단말 이름들의 집합)
      * 시작 기호 S 에는 항상 '$'가 포함됩니다.
    """
    nullable: Set[str]
    first: Dict[str, Set[str]]
    follow: Dict[str, Set[str]]


def compute_nullable_first_follow(bnf: BNF, sym: SymbolTable) -> FFResult:
    """
    compute_nullable_first_follow
    =============================
    BNF 형태의 문법에 대해 NULLABLE/FIRST/FOLLOW 집합을 계산합니다.
    반환 값은 **모두 이름 기반**(문자열)입니다. (테이블 생성 시 ID로 치환하면 됩니다)

    알고리즘 개요
    ------------
    1) NULLABLE
       - ε-프로덕션(A -> ε)이 있으면 A를 nullable에 추가
       - A -> X1 X2 ... Xn 에서 모든 Xi가 nullable이면 A도 nullable
       - 더 이상 변화가 없을 때까지 반복

    2) FIRST
       - 단말 a: FIRST(a) = { a }
       - 비단말 A: 모든 프로덕션 A -> α 에 대해 FIRST(α)를 합집합
       - FIRST(α): α의 각 심볼을 왼쪽부터 훑으며
           - 단말이면 그 단말을 추가하고 중단
           - 비단말이면 FIRST(nonterm)를 추가(단 ε 제외).
           - 해당 심볼이 nullable이면 다음 심볼로 진행, 아니면 중단
         만약 전체 α가 nullable이면 FIRST(α)는 ε를 포함한다고 볼 수 있으나,
         여기서는 별도의 ε 기호를 넣지 않고 **nullable**이 이를 대변합니다.

    3) FOLLOW
       - FOLLOW(start) 에 '$' 추가
       - 모든 프로덕션 A -> X1 X2 ... Xn 에 대해, 오른쪽에서 왼쪽으로 훑으며
           - trailer := FOLLOW(A)로 시작
           - Xi가 비단말이면 FOLLOW(Xi)에 trailer를 더함
           - 이후 trailer := FIRST(Xi) ∪ (Xi가 nullable이면 trailer 포함)
         변화가 없을 때까지 반복
    """
    terms = set(bnf.terms)
    nonterms = set(bnf.nonterms)

    # ---------- 0) 준비: first/nullable 초기화 ----------
    nullable: Set[str] = set()
    first: Dict[str, Set[str]] = {}

    # 단말의 FIRST는 자기 자신
    for t in terms:
        first[t] = {t}
    # 비단말은 일단 빈 집합
    for A in nonterms:
        first[A] = set()

    # ---------- 1) NULLABLE 고정점 ----------
    changed = True
    while changed:
        changed = False
        for p in bnf.prods:
            A, alpha = p.lhs, p.rhs
            if not alpha:
                if A not in nullable:
                    nullable.add(A)
                    changed = True
                continue
            # alpha의 모든 심볼이 nullable인지 검사
            all_null = True
            for X in alpha:
                if X in terms:
                    all_null = False
                    break
                if X not in nullable:
                    all_null = False
                    break
            if all_null and A not in nullable:
                nullable.add(A)
                changed = True

    # ---------- 2) FIRST 고정점 ----------
    def first_of_sequence(seq: List[str]) -> Tuple[Set[str], bool]:
        """
        주어진 심볼 시퀀스 seq에 대한 FIRST 집합(단말 이름의 집합)과
        'seq 자체가 nullable인지' 여부를 반환합니다.
        """
        out: Set[str] = set()
        if not seq:
            return out, True  # ε
        all_null = True
        for X in seq:
            if X in terms:
                out.add(X)
                all_null = False
                break
            # 비단말
            out |= first[X]
            # X가 nullable이 아니면 더 진행 불가
            if X not in nullable:
                all_null = False
                break
        return out, all_null

    changed = True
    while changed:
        changed = False
        for p in bnf.prods:
            A, alpha = p.lhs, p.rhs
            f_alpha, _ = first_of_sequence(alpha)
            before = len(first[A])
            first[A] |= f_alpha
            if len(first[A]) != before:
                changed = True

    # ---------- 3) FOLLOW 고정점 ----------
    follow: Dict[str, Set[str]] = {A: set() for A in nonterms}
    follow[bnf.start].add("$")  # EOF

    changed = True
    while changed:
        changed = False
        for p in bnf.prods:
            A, alpha = p.lhs, p.rhs
            trailer: Set[str] = set(follow[A])  # 오른쪽에서 왼쪽으로 전파될 집합
            for X in reversed(alpha):
                if X in nonterms:
                    # FOLLOW(X) ⊇ trailer
                    before = len(follow[X])
                    follow[X] |= trailer
                    if len(follow[X]) != before:
                        changed = True
                    # trailer 갱신: FIRST(X) ∪ (nullable(X) ? trailer : ∅)
                    trailer = set(first[X]) | (trailer if X in nullable else set())
                else:
                    # X가 단말이면 trailer = FIRST(X) = {X}
                    trailer = {X}

    return FFResult(nullable=nullable, first=first, follow=follow)
