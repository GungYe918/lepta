# lepta/grammar/transform.py
"""EBNF(?,*,+)를 BNF로 변환하고 Production 리스트로 리턴"""

from __future__     import annotations
from dataclasses    import dataclass, field
from typing         import List, Optional, Set, Tuple
from .ast           import *
from .templates.precedence_templates import TEMPLATES


@dataclass 
class Production:
    """
    BNF 프로덕션 1개.
    - lhs: 좌변 비단말 이름
    - rhs: 우변 심볼 이름 리스트(터미널/비단말)
    - prec_label: 이 프로덕션에 부여된 우선순위 라벨(없으면 None)
    """
    lhs: str
    rhs: List[str]      # ε는 빈 리스트([])로 표현
    prec_label: Optional[str] = None


@dataclass
class BNF:
    start: str
    prods: List[Production]
    terms: List[str]
    nonterms: List[str]
    # 우선순위 선언(템플릿 전개 완료본)
    precedences: List[PrecedenceDecl] = field(default_factory=list)


def _expand_precedence(g: Grammar) -> List[PrecedenceDecl]:
    """
    문법의 precedence 템플릿 및 개별 선언을 합쳐서 하나의 리스트로 만듭니다.
    아래에 적을수록 '더 강한 레벨' 이 됩니다.
    """
    out: List[PrecedenceDecl] = []
    # 템플릿 전개
    for use in getattr(g, "decl_prec_templates", []):
        templ = TEMPLATES.get(use.name)
        if templ is None:
            raise SyntaxError(f"Unknown precedence template: {use.name}")
        for assoc, labels in templ:
            out.append(PrecedenceDecl(assoc=assoc, labels=list(labels)))
    # 사용자 직접 선언
    for d in getattr(g, "decl_precedences", []):
        out.append(d)
    return out


class _Lowering:
    def __init__(self, g: Grammar) :
        self.g = g
        self.prods: List[Production] = []
        self.terms: Set[str] = set()
        self.nonterms: Set[str] = set()
        self._grp_id = 0
        self._rep_id = 0
        self._opt_id = 0
        # 토큰 이름 목록(단말 취급)
        self.token_terms: Set[str] = {t.name for t in g.decl_tokens}
        # 키워드 리터럴(단말)
        for kw in g.decl_keywords:
            self.terms.add(kw.lexeme)

    # 새 비단말 이름
    def _new_grp(self) -> str:
        self._grp_id += 1
        name = f"__grp{self._grp_id}"
        self.nonterms.add(name)
        return name

    def _new_rep(self) -> str:
        self._rep_id += 1
        name = f"__rep{self._rep_id}"
        self.nonterms.add(name)
        return name

    def _new_opt(self) -> str:
        self._opt_id += 1
        name = f"__opt{self._opt_id}"
        self.nonterms.add(name)
        return name
    
    # 리터럴/토큰/비단말 구분하여 심볼 문자열 반환, 집합 갱신
    def _sym_from_name(self, ident: str) -> str:
        if ident in self.token_terms:
            self.terms.add(ident)
            return ident  # 단말
        else:
            self.nonterms.add(ident)
            return ident  # 비단말
        
    
    def _syms_from_atom_base(self, atom: Atom) -> List[str]:
        node = atom.node
        if isinstance(node, Name):
            return [self._sym_from_name(node.ident)]
        elif isinstance(node, Lit):
            self.terms.add(node.text)
            return [node.text]
        elif isinstance(node, Group):
            # 그룹을 새 비단말로 치환하고 그 비단말 반환
            grp_name = self._new_grp()
            self._lower_expr_into(grp_name, node.expr)
            return [grp_name]
        else:
            raise TypeError("unknown Atom.node")
        
    
    def _lower_seq_atoms(self, atoms: List[Atom]) -> List[str]:
        """시퀀스 내 원자들을 전개하여 RHS 심볼 리스트로 반환."""
        rhs: List[str] = []
        for a in atoms:
            base_syms = self._syms_from_atom_base(a)  # 1개 이상일 수 있음(그룹)
            if a.suffix == Suffix.NONE:
                rhs.extend(base_syms)
            elif a.suffix == Suffix.OPT:
                opt = self._new_opt()
                # opt -> ε | base
                self.prods.append(Production(opt, []))
                self.prods.append(Production(opt, base_syms.copy()))
                rhs.append(opt)
            elif a.suffix == Suffix.STAR:
                rep = self._new_rep()
                # rep -> ε | base rep   (우측 재귀)
                self.prods.append(Production(rep, []))
                self.prods.append(Production(rep, base_syms + [rep]))
                rhs.append(rep)
            elif a.suffix == Suffix.PLUS:
                rep = self._new_rep()
                # rep -> ε | base rep
                self.prods.append(Production(rep, []))
                self.prods.append(Production(rep, base_syms + [rep]))
                # PLUS는 최소 1회: base + rep
                rhs.extend(base_syms)
                rhs.append(rep)
            else:
                raise ValueError(f"unknown suffix: {a.suffix}")
        return rhs
    
    def _lower_expr_into(self, lhs: str, expr: Expr) -> None:
        self.nonterms.add(lhs)
        # 각 대안(Seq)을 하나의 프로덕션으로
        for seq in expr.alts:
            rhs = self._lower_seq_atoms(seq.items)
            # 원본 Seq.prec를 최종 프로덕션의 prec_label로 전파
            self.prods.append(Production(lhs, rhs, prec_label=seq.prec))

    def lower(self) -> BNF:
        # 명시적 start 없으면 첫 규칙
        start = self.g.start or (self.g.rules[0].name if self.g.rules else "Start")
        # 원본 규칙 전개
        for r in self.g.rules:
            self._lower_expr_into(r.name, r.expr)
        # terms/nonterms 최종 보정: 토큰 선언 자체도 단말 후보
        self.terms |= self.token_terms

        # precedence 템플릿/직접 선언 전개
        precedences = _expand_precedence(self.g)

        # BNF 스키마에 맞게 list로 변환하여 반환
        return BNF(
            start=start,
            prods=self.prods,
            terms=sorted(self.terms),
            nonterms=sorted(self.nonterms),
            precedences=precedences,
        )


def to_bnf(g: Grammar) -> BNF:
    """Grammar(AST) → BNF(프로덕션 목록, 단말/비단말 집합)"""
    return _Lowering(g).lower()
