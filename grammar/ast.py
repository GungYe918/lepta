# lepta/grammar/ast.py
"""Grammar AST
- TokenDecl: %token PATTERN
- IgnoreDecl: %ignore PATTERN
- Expr/Seq/Atom: EBNF 표현을 그대로 보존(?,*,+ 포함)
"""

from __future__     import annotations
from dataclasses    import dataclass, field
from typing         import List, Optional, Union, Dict, Tuple

@dataclass
class Span:
    start: int
    end: int
    line: int
    col: int

@dataclass 
class TokenDecl:
    name: str
    pattern: str    # 원본 정규식 문자열
    flags: str = ""
    span: Optional[Span] = None

@dataclass
class IgnoreDecl:
    pattern: str
    flags: str = ""
    span: Optional[Span] = None


@dataclass
class KeywordDecl:
    lexeme: str
    span: Optional[Span] = None

@dataclass 
class PrecedenceDecl:
    """
    우선순위 선언 한 줄: %left/%right/%nonassoc label...
    - assoc: 'left' | 'right' | 'nonassoc'
    - labels: 토큰명(IDENT) 또는 리터럴 문자열('"+"')의 리스트
    """
    assoc: str
    labels: List[str]

@dataclass
class PrecedenceTemplateUse:
    """
    %use precedence <TEMPLATE_NAME>; 한 줄을 표현.
    - name: 템플릿 이름
    """
    name: str

@dataclass
class PrecLevel:
    """
    해석(resolved)된 우선순위 레벨 1개.
    - assoc : 'left' | 'right' | 'nonassoc'
    - labels: 이 레벨에 속한 토큰 라벨들(토큰명 또는 리터럴)
    """
    assoc: str
    labels: List[str]

@dataclass
class ResolvedPrecedence:
    """
    변환(transform) 단계에서 만들어지는 우선순위 테이블의 최종 형태.
    - levels           : 낮은 우선순위 → 높은 우선순위 순서의 레벨 목록
    - label_to_level   : 라벨(토큰명/리터럴) -> 레벨 인덱스(0..len(levels)-1)
    - label_assoc      : 라벨 -> 결합성('left'|'right'|'nonassoc')
    """
    levels: List[PrecLevel] = field(default_factory=list)
    label_to_level: Dict[str, int] = field(default_factory=dict)
    label_assoc: Dict[str, str] = field(default_factory=dict)

class Suffix:
    NONE = "none"
    OPT  = "opt"
    STAR = "star"
    PLUS = "plus"

@dataclass
class Name:
    ident: str
    span: Optional[Span] = None

@dataclass
class Lit:
    text: str
    span: Optional[Span] = None

@dataclass
class Group:
    expr: "Expr"
    span: Optional[Span] = None

@dataclass
class PegExprRef:
    """RHS용 국소 PEG 임베딩: @peg(Block.Rule)"""
    block: str
    rule: str
    span: Optional[Span] = None


AtomKind = Union[Name, Lit, Group, PegExprRef]

# EBNF 표현 구조

@dataclass
class Atom:
    node: AtomKind
    suffix: str = Suffix.NONE
    span: Optional[Span] = None

@dataclass
class Seq:
    """
    대안(alt) 하나의 시퀀스.
    - items: Atom 리스트
    - prec : 이 시퀀스에 부여된 %prec 라벨(없으면 None)
    """
    items: List[Atom]
    prec: Optional[str] = None


@dataclass
class Expr:
    alts: List[Seq]

@dataclass
class Rule:
    name: str
    expr: Expr
    span: Optional[Span] = None

# ====== PEG 통합용 노드

@dataclass
class PegBlockDecl:
    """%peg NAME { ... };  — PEG 블록 원문을 그대로 보존"""
    name: str
    src: str    # 중괄호 내부 원문(문자 그대로)

@dataclass
class PegTokenDecl:
    """%token NAME %peg(Block.Rule) [trigger='x'];  — PEG 기반 토큰"""
    name: str
    peg_ref: Tuple[str, str]           # (block_name, rule_name)
    trigger: Optional[str] = None      # 단일 문자 트리거(선택)


@dataclass
class Grammar:
    # 선언(Decl) 섹션
    decl_tokens: List[TokenDecl] = field(default_factory=list)
    decl_ignores: List[IgnoreDecl] = field(default_factory=list)
    decl_keywords: List[KeywordDecl] = field(default_factory=list)

    # PEG 선언
    decl_peg_blocks: List[PegBlockDecl] = field(default_factory=list)
    decl_peg_tokens: List[PegTokenDecl]  = field(default_factory=list)
    
    # 우선순위 선언
    decl_precedences: List[PrecedenceDecl] = field(default_factory=list)
    decl_prec_templates: List[PrecedenceTemplateUse] = field(default_factory=list)

    # 규칙 섹션
    rules: List[Rule] = field(default_factory=list)
    start: Optional[str] = None

    # 변환(transform) 단계에서 완성되는 우선순위 테이블(해석 결과)
    resolved_prec: Optional[ResolvedPrecedence] = None

    # 전역 옵션/플래그(예: parser 모드, 디버그 등)
    options: Dict[str, str] = field(default_factory=dict)

    # ===== 오류 복구(전역) =====
    # - recover_mode: None|"off"|"panic" (기본 None → off 취급)
    # - sync_labels : 전역 동기화 토큰 라벨 목록(예: [")", ";", "EOF"])
    recover_mode: Optional[str] = None
    sync_labels: List[str] = field(default_factory=list)
