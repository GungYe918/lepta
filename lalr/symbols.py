"""심볼에 정수 ID를 부여해 테이블에서 사용하기 쉽게 합니다."""
from __future__     import annotations
from dataclasses    import dataclass
from typing         import Dict, List, Set, Optional

@dataclass
class SymbolTable:
    """
    SymbolTable
    ===========
    단말/비단말 **이름 ↔ 정수 ID 매핑**을 관리하는 테이블입니다.
    Rasca의 모든 내부 단계(FIRST/FOLLOW, 테이블 생성, 런타임)에서 동일한 심볼 ID를 쓰기 위해
    이 테이블을 '고정(freeze)'한 뒤 사용합니다.

    설계 원칙
    --------
    - 단말(terminal)과 비단말(nonterminal)의 ID 영역을 **분리**합니다.
      - 단말 ID: 0 .. T-1, **EOF('$')는 항상 0**번으로 예약합니다.
      - 비단말 ID: T .. T+N-1
    - freeze() 이후에는 이름↔ID 매핑이 **불변**입니다.
    - 이름은 transform 단계(BNF)에서 쓰이는 **문자열 심볼 이름**을 그대로 사용합니다.
      - 예: 단말: "NUMBER", "+", "(", ")" / 비단말: "Expr", "__rep1" 등

    주요 속성/메서드
    ----------------
    - freeze(terms, nonterms, start):
        주어진 단말/비단말 집합과 시작기호를 바탕으로 테이블을 확정합니다.
        '$'(EOF)가 단말 집합에 없으면 자동으로 추가합니다.
    - id_of(name) / name_of(id):
        이름 ↔ ID 변환
    - is_term_id(id) / is_nonterm_id(id):
        주어진 ID가 단말/비단말인지 판별
    - eof_id:
        EOF('$')의 ID를 반환(항상 0)
    """

    # 내부 저장소
    _name_to_id: Dict[str, int] = None
    _id_to_name: List[str] = None
    _term_count: int = 0
    _nonterm_count: int = 0
    _frozen: bool = False
    _start: Optional[str] = None

    def freeze(self, terms: Set[str], nonterms: Set[str], start: str) -> None:
        """
        단말/비단말 집합으로 심볼 테이블을 '고정'합니다.
        - '$' (EOF)를 단말의 **ID=0**으로 강제 배정합니다.
        - 나머지 단말/비단말은 알파벳 정렬(디버깅 편의)로 ID를 배정합니다.
        """

        if self._frozen:
            return 
        
        # '$' EOF 단말(terminal) 확보
        terms = set(terms)
        terms.add("$")


        # 단말/비단말 ID 부여
        term_names = sorted(terms)
        # '$'를 항상 0번째로 오게 재배열
        if term_names[0] != "$":
            term_names.remove("$")
            term_names.insert(0, "$")

        nonterm_names = sorted(nonterms)

        
        # 이름 → ID, ID → 이름
        self._name_to_id = {}
        self._id_to_name = []


        # 단말 먼저
        for i, nm in enumerate(term_names):
            self._name_to_id[nm] = i
            self._id_to_name.append(nm)
        self._term_count = len(term_names)

        # 비단말
        for j, nm in enumerate(nonterm_names):
            idx = self._term_count + j
            self._name_to_id[nm] = idx
            self._id_to_name.append(nm)
        self._nonterm_count = len(nonterm_names)

        self._start = start
        self._frozen = True

    
    # ----- 조회 / 유틸 -----
    def id_of(self, name:str) -> int:
        """심볼 이름을 ID로 변환합니다. 존재하지 않으면 KeyError."""
        return self._name_to_id[name]
    
    def name_of(self, id_: int) -> str:
        """심볼 ID를 이름으로 변환합니다. 범위를 벗어나면 IndexError."""
        return self._id_to_name[id_]

    def is_term_id(self, id_: int) -> bool:
        """해당 ID가 단말인지 여부."""
        return 0 <= id_ < self._term_count

    def is_nonterm_id(self, id_: int) -> bool:
        """해당 ID가 비단말인지 여부."""
        return self._term_count <= id_ < (self._term_count + self._nonterm_count)
    
    @property
    def eof_id(self) -> int:
        """EOF 단말('$')의 ID (항상 0)."""
        return 0

    @property
    def term_count(self) -> int:
        return self._term_count

    @property
    def nonterm_count(self) -> int:
        return self._nonterm_count

    @property
    def start(self) -> str:
        return self._start

    @property
    def start_id(self) -> int:
        """시작 비단말의 ID."""
        return self.id_of(self._start)
    
    def __repr__(self) -> str:
        terms = [self._id_to_name[i] for i in range(self._term_count)]
        nonterms = [self._id_to_name[i] for i in range(self._term_count, self._term_count + self._nonterm_count)]
        return f"SymbolTable(terms={terms}, nonterms={nonterms}, start={self._start})"

    
