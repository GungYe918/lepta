from __future__ import annotations
from pathlib import Path
from typing import List
from .loader import load_grammar_text
from .parser import parse_grammar
from .transform import to_bnf, Production, BNF
from ..lalr.symbols import SymbolTable
from ..lalr.first_follow import compute_nullable_first_follow
from ..lalr.items import build_slr_tables
from ..lalr.runtime import parse_string
from ..lex import Lexer
from ..codegen.ir import build_ir
from ..codegen.emit_rs import emit_rs_to_string

EXPR = Path("lepta/tests/grammar_test/expr.g")

def _format_production(p: Production) -> str:
    rhs = " ".join(p.rhs) if p.rhs else "ε"
    return f"{p.lhs} -> {rhs}"

def _print_bnf(b: BNF) -> None:
    print(f"\n[BNF]")
    print(f"Start: {b.start}")
    # 정렬 출력(디버깅 용이)
    by_lhs = {}
    for prod in b.prods:
        by_lhs.setdefault(prod.lhs, []).append(prod)
    for lhs in sorted(by_lhs):
        for p in sorted(by_lhs[lhs], key=lambda x: " ".join(x.rhs)):
            print(_format_production(p))
    print("\n[Terminals]")
    print(", ".join(sorted(b.terms)))
    print("\n[Nonterminals]")
    print(", ".join(sorted(b.nonterms)))

def _print_first_follow(bnf):
    """FIRST/FOLLOW/NULLABLE을 계산해 보기 좋게 출력합니다."""
    sym = SymbolTable()
    sym.freeze(bnf.terms, bnf.nonterms, bnf.start)

    ff = compute_nullable_first_follow(bnf, sym)

    print("\n[NULLABLE]")
    print(", ".join(sorted(ff.nullable)) if ff.nullable else "(none)")

    print("\n[FIRST(nonterminals)]")
    for A in sorted(bnf.nonterms):
        items = sorted(ff.first[A])
        print(f"{A:>10} : {{{', '.join(items)}}}")

    print("\n[FOLLOW(nonterminals)]")
    for A in sorted(bnf.nonterms):
        items = sorted(ff.follow[A])
        print(f"{A:>10} : {{{', '.join(items)}}}")

def _print_slr_tables(bnf):
    """
    BNF와 심볼테이블, FOLLOW를 이용해 SLR(1) 테이블을 생성하고
    상태 수/충돌/일부 액션을 요약 출력합니다.
    """
    sym = SymbolTable()
    sym.freeze(bnf.terms, bnf.nonterms, bnf.start)

    ff = compute_nullable_first_follow(bnf, sym)
    tbl = build_slr_tables(bnf, sym, ff.follow)

    print("\n[SLR(1) Tables]")
    print(f"States: {tbl.n_states}")
    print(f"Conflicts: {len(tbl.conflicts)}")
    if tbl.conflicts:
        print(tbl.pretty_conflicts(sym.name_of))

    # 상태 0에서의 액션 샘플(디버깅)
    sample_terms = [t for t in ["NUMBER", "(", "+", "*", ")", "$"] if t in bnf.terms or t == "$"]
    print("\n[ACTIONS in state 0]")
    for t in sample_terms:
        tid = sym.id_of(t) if t != "$" else sym.eof_id
        act = tbl.action.get((0, tid))
        print(f"  on {t:>6} -> {act}")

    # 간단한 accept 유무 검사
    has_accept = any(k for k, v in tbl.action.items() if v[0] == 'acc')
    print(f"\nAccept present: {has_accept}")

def _run_runtime_smoke(ast, bnf):
    """렌타임 스택 머신으로 간단한 입력을 파싱해 accept/에러를 확인."""
    from ..lalr.symbols import SymbolTable
    from ..lalr.first_follow import compute_nullable_first_follow
    from ..lalr.items import build_slr_tables

    # 심볼 테이블/테이블 구축
    sym = SymbolTable()
    sym.freeze(bnf.terms, bnf.nonterms, bnf.start)
    ff = compute_nullable_first_follow(bnf, sym)
    tbl = build_slr_tables(bnf, sym, ff.follow)

    # 렉서 & 입력
    lx = Lexer(ast)
    ok_inputs = [
        "1+2*3+(4*5)",
        "42",
        "(1+2)*3",
        "1+2+3",
        "((7))",
    ]
    bad_inputs = [
        "1+*2",
        "(",
        "1+(2*3",
    ]

    print("\n[Runtime OK cases]")
    for s in ok_inputs:
        try:
            res = parse_string(s, tbl, sym, lx)
            print(f"  {s!r} -> {res}")
        except SyntaxError as e:
            print(f"  {s!r} -> ERROR\n{e}")
            raise

    print("\n[Runtime ERROR cases]")
    for s in bad_inputs:
        try:
            res = parse_string(s, tbl, sym, lx)
            print(f"  {s!r} -> {res} (unexpected)")
        except SyntaxError as e:
            print(f"  {s!r} -> SyntaxError: \n{e}")

def _print_emit_rs_preview(ast, bnf):
    """SLR 테이블로 IR을 만들고, Rust 단일파일 소스를 문자열로 생성한 뒤 프리뷰한다."""
    sym = SymbolTable()
    sym.freeze(bnf.terms, bnf.nonterms, bnf.start)
    ff = compute_nullable_first_follow(bnf, sym)
    tbl = build_slr_tables(bnf, sym, ff.follow)

    ir = build_ir(sym, tbl)
    rs_src = emit_rs_to_string(ir, module_name="expr_parser")

    print("\n[Emit Rust Preview]")
    print(f"Generated Rust source length: {len(rs_src)} bytes")
    print("--- snippet ---")
    lines = rs_src.splitlines()
    for i in range(min(20, len(lines))):
        print(lines[i])
    print("... (snip) ...")

    # 파일로 저장하려면(원하면 주석 해제):
    # out_path = 'tests/tmp/parser.rs'
    # with open(out_path, 'w', encoding='utf-8') as f:
    #     f.write(rs_src)
    # print(f"Wrote: {out_path}")


def main() -> None:
    try:
        text = load_grammar_text(str(EXPR))
        ast = parse_grammar(text)
        print("[AST]")
        print(ast)
        bnf = to_bnf(ast)
        _print_bnf(bnf)
        _print_first_follow(bnf)
        _print_slr_tables(bnf)
        _run_runtime_smoke(ast, bnf)
        _print_emit_rs_preview(ast, bnf)
    except SyntaxError as e:
        # 친절한 메시지만 출력(Traceback 숨김)
        print(str(e))


if __name__ == "__main__":
    main()
