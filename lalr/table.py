# table.py
"""
나중에 여기에 설명 작성
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Optional


@dataclass
class Tables:
    """
    Tables
    ======
    SLR(1) 파서 테이블과 디버그 정보를 담는 컨테이너.

    필드
    ----
    - action: (state:int, term_id:int) -> ('s', next_state) | ('r', prod_idx) | ('acc', 0)
        * 's' : shift
        * 'r' : reduce (prod_idx는 BNF.prods(증강 포함)에서의 인덱스)
        * 'acc': accept
    - goto  : (state:int, nonterm_id:int) -> next_state
    - start_state: 증강문법의 시작 상태(I0)의 상태 번호
    - n_states   : 전체 상태 수
    - prod_lhs_ids: 각 프로덕션의 LHS 심볼 ID (length = #prods, 증강 포함)
    - prod_rhs_len: 각 프로덕션의 RHS 길이 (length = #prods, 증강 포함)
    - state_items : 디버깅용. 상태별 아이템의 문자열 표현 리스트
    - conflicts  : 충돌 보고 리스트. (state, symbol_id, (kind1, kind2))
        예: ('shift','reduce'), ('reduce','reduce'), ('accept','reduce')

    사용
    ----
    - 런타임 파서는 action/goto 를 조회하여 파싱합니다.
    - 에러 메시지의 expected-set은 특정 상태 s에서 action[(s, term)]가 존재하는 term들의 집합으로 계산.
    - pretty_conflicts()는 심볼 이름 복원을 위해 id->name 콜백을 받아 사람이 읽기 좋은 내용을 만듭니다.
    """
    action: Dict[Tuple[int, int], Tuple[str, int]]
    goto: Dict[Tuple[int, int], int]
    start_state: int
    n_states: int
    prod_lhs_ids: List[int]
    prod_rhs_len: List[int]
    state_items: List[List[str]]
    conflicts: List[Tuple[int, int, Tuple[str, str]]]

    def pretty_conflicts(self, id_to_name: Callable[[int], str]) -> str:
        """
        충돌 목록을 사람이 읽기 좋은 문자열로 변환합니다.

        Parameters
        ----------
        id_to_name : Callable[[int], str]
            심볼 ID를 심볼 이름 문자열로 변환하는 콜백(SymbolTable.name_of 등).

        Returns
        -------
        str
            줄바꿈으로 구분된 충돌 리포트. 충돌이 없으면 '(no conflicts)' 반환.
        """
        if not self.conflicts:
            return "(no conflicts)"
        lines: List[str] = []
        for st, sym, kinds in self.conflicts:
            try:
                sym_name = id_to_name(sym)
            except Exception:
                sym_name = f"#{sym}"
            lines.append(f"state {st}, on {sym_name}: {kinds[0]} / {kinds[1]}")
        return "\n".join(lines)
