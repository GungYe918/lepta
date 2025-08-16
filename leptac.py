# lepta/leptac.py
"""leptac – lepta CLI

사용 예)
    $ python -m lepta.leptac check tests/grammar_test/expr.g -D --parser lalr
    $ python -m lepta.leptac build tests/grammar_test/expr.g --parser lalr --lang rust -o tests/tmp/parser.rs -D
    $ python -m lepta.leptac build tests/grammar_test/expr.g --parser lalr --lang c    -o tests/tmp/parser.c  -D

기능
----
- check : 문법을 읽어 파이프라인(AST→BNF→FIRST/FOLLOW→(SLR|LALR)) 검증 및 요약 출력
- build : 문법을 읽어 타깃 언어 코드(현재 Rust/C)로 방출

디버그 모드(-D/--debug)를 켜면 AST/BNF/테이블 요약과 충돌 리포트를 출력합니다.
"""

from __future__ import annotations
import argparse
import pathlib
import re
import sys
from typing import Optional

# ------------------------------
# 헬퍼
# ------------------------------

def _eprint(*args, **kw) -> None:
    print(*args, file=sys.stderr, **kw)


def _sanitize_rs_mod_name(name: str) -> str:
    """Rust 모듈명 규칙에 맞춰 파일명을 식별자로 변환.
    C의 emit에서도 단순 주석/식별자 문자열로만 사용한다."""
    stem = pathlib.Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    if not stem or not re.match(r"[A-Za-z_]", stem[0]):
        stem = "parser_" + (stem or "out")
    return stem

# ------------------------------
# 파이프라인 로딩
# ------------------------------

def _load_pipeline(grammar_path: str, debug: bool, parser_kind: str):
    """
    .g 파일을 읽어 AST→BNF→심볼→FIRST/FOLLOW→(SLR|LALR) 테이블까지 생성.
    parser_kind: 'slr' or 'lalr'
    """
    from .grammar.loader import load_grammar_text
    from .grammar.parser import parse_grammar
    from .grammar.transform import to_bnf
    from .lalr.symbols import SymbolTable
    from .lalr.first_follow import compute_nullable_first_follow
    from .lalr.items import build_slr_tables, build_lalr_tables

    src = load_grammar_text(grammar_path)
    g = parse_grammar(src)
    if debug: _eprint("[DEBUG] AST ready")

    bnf = to_bnf(g)
    if debug: _eprint("[DEBUG] BNF ready | terms=%d nonterms=%d rules=%d" %
                      (len(bnf.terms), len(bnf.nonterms), len(bnf.prods)))

    sym = SymbolTable()
    sym.freeze(bnf.terms, bnf.nonterms, bnf.start)
    if debug: _eprint("[DEBUG] SymbolTable frozen | term=%d nonterm=%d start=%s" %
                      (sym.term_count, sym.nonterm_count, bnf.start))

    ff = compute_nullable_first_follow(bnf, sym)
    if debug: _eprint("[DEBUG] FIRST/FOLLOW/NULLABLE computed")

    if parser_kind == "slr":
        tbl = build_slr_tables(bnf, sym, ff.follow)
        if debug: _eprint("[DEBUG] SLR tables built | states=%d conflicts=%d" %
                          (tbl.n_states, len(tbl.conflicts)))
    else:
        tbl = build_lalr_tables(bnf, sym)
        if debug: _eprint("[DEBUG] LALR tables built | states=%d conflicts=%d" %
                          (tbl.n_states, len(tbl.conflicts)))

    return g, bnf, sym, ff, tbl

# ------------------------------
# 디버그 출력 헬퍼
# ------------------------------

def _print_ast(g) -> None:
    _eprint("\n[AST]\n" + repr(g))

def _print_bnf_summary(bnf) -> None:
    _eprint("\n[BNF]")
    _eprint(f"Start: {bnf.start}")
    _eprint("Terminals:")
    _eprint("  " + ", ".join(bnf.terms))
    _eprint("Nonterminals:")
    _eprint("  " + ", ".join(bnf.nonterms))
    _eprint(f"Productions: {len(bnf.prods)}")


def _print_tables_summary(tbl, sym) -> None:
    _eprint("\n[Parsing Tables]")
    _eprint(f"States: {tbl.n_states}")
    _eprint(f"Conflicts: {len(tbl.conflicts)}")
    if tbl.conflicts:
        _eprint("\n[Conflicts Detail]")
        _eprint(tbl.pretty_conflicts(sym.name_of))
    _eprint("\n[State 0 items]")
    if tbl.state_items:
        for line in tbl.state_items[0]:
            _eprint("  " + line)

# ------------------------------
# 커맨드 구현
# ------------------------------

def cmd_check(args) -> int:
    try:
        g, bnf, sym, ff, tbl = _load_pipeline(args.file, debug=args.debug, parser_kind=args.parser)
    except SyntaxError as e:
        _eprint("[SYNTAX ERROR]")
        _eprint(str(e))
        return 2
    except Exception as e:
        _eprint("[ERROR]", type(e).__name__, str(e))
        return 2

    if args.debug:
        _print_ast(g)
        _print_bnf_summary(bnf)
        _print_tables_summary(tbl, sym)

    print(f"[CHECK OK] parser={args.parser} states={tbl.n_states} prods={len(tbl.prod_rhs_len)} conflicts={len(tbl.conflicts)}")
    return 0


def cmd_build(args) -> int:
    try:
        g, bnf, sym, ff, tbl = _load_pipeline(args.file, debug=args.debug, parser_kind=args.parser)
    except SyntaxError as e:
        _eprint("[SYNTAX ERROR]")
        _eprint(str(e))
        return 2
    except Exception as e:
        _eprint("[ERROR]", type(e).__name__, str(e))
        return 2

    if args.debug:
        _print_ast(g)
        _print_bnf_summary(bnf)
        _print_tables_summary(tbl, sym)

    if tbl.conflicts:
        _eprint("[WARN] Conflicts present; continuing with shift-preferred resolution.")
        _eprint(tbl.pretty_conflicts(sym.name_of))

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- IR 생성  ----
    from .codegen.ir import build_ir
    ir = build_ir(sym, tbl, g) 

    if args.lang == "rust":
        from .codegen.emit_rs import emit_rs_to_string
        module_name = args.module or _sanitize_rs_mod_name(out_path.name)
        src = emit_rs_to_string(
            ir,
            module_name=module_name,
            lexer_ast=(g if args.with_lexer else None),
        )
        out_path.write_text(src, encoding="utf-8")
        print(f"[EMIT] parser={args.parser} lang=rust -> {out_path}")
        if args.debug:
            _eprint(f"[DEBUG] module={module_name} bytes={len(src)}")
        return 0

    elif args.lang == "c":
        from .codegen.emit_c import emit_c_to_string
        module_name = args.module or _sanitize_rs_mod_name(out_path.name)
        src = emit_c_to_string(ir, module_name=module_name)
        out_path.write_text(src, encoding="utf-8")
        print(f"[EMIT] parser={args.parser} lang=c -> {out_path}")
        if args.debug:
            _eprint(f"[DEBUG] module={module_name} bytes={len(src)}")
        return 0

    else:
        _eprint(f"[ERROR] Unsupported language: {args.lang}")
        return 2
    

def cmd_lex(args) -> int:
    """문법으로 간단 토크나이즈를 수행해 결과를 표준출력으로 보여줍니다."""
    try:
        from .grammar.loader import load_grammar_text
        from .grammar.parser import parse_grammar
        from .lex import SimpleLexer
        src = load_grammar_text(args.file)
        g = parse_grammar(src)
        lx = SimpleLexer.from_grammar(g)
        if args.text is not None:
            text = args.text
        else:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        lx.reset(text)
        i = 0
        while True:
            tok = lx.next()
            if tok is None:
                break
            print(f"{i:03d}: {tok.type:<12} {tok.text!r}  @{tok.line}:{tok.col}")
            i += 1
        return 0
    except SyntaxError as e:
        _eprint("[LEX ERROR]", str(e))
        return 2
    except Exception as e:
        _eprint("[ERROR]", type(e).__name__, str(e))
        return 2


# ------------------------------
# 엔트리포인트
# ------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="leptac", description="lepta parser generator CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="문법을 검사하고 테이블을 생성해 충돌 유무를 확인합니다")
    p_check.add_argument("file", help=".g 문법 파일")
    p_check.add_argument("--parser", choices=["slr", "lalr"], default="lalr", help="파서 테이블 방식")
    p_check.add_argument("-D", "--debug", action="store_true", help="디버그 정보를 상세 출력")
    p_check.set_defaults(func=cmd_check)

    p_build = sub.add_parser("build", help="타깃 언어(C/Rust)용 파서를 생성합니다")
    p_build.add_argument("file", help=".g 문법 파일")
    p_build.add_argument("--parser", choices=["slr", "lalr"], default="lalr", help="파서 테이블 방식")
    p_build.add_argument("--lang", choices=["c", "rust"], required=True, help="타깃 언어")
    p_build.add_argument("-o", "--output", required=True, help="출력 파일 경로")
    p_build.add_argument("-m", "--module", help="(Rust/C) 모듈/식별자 이름(미지정시 출력 파일명에서 유도)")
    p_build.add_argument("--with-lexer", action="store_true", help="(Rust) 내장 간단 렉서 포함")
    p_build.add_argument("-D", "--debug", action="store_true", help="디버그 정보를 상세 출력")
    p_build.set_defaults(func=cmd_build)

    p_lex = sub.add_parser("lex", help="문법을 이용해 입력 텍스트를 토크나이즈합니다")
    p_lex.add_argument("file", help=".g 문법 파일")
    src_group = p_lex.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--text", help="직접 입력 텍스트")
    src_group.add_argument("--input", help="입력 텍스트 파일 경로")
    p_lex.set_defaults(func=cmd_lex)

    args = ap.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    sys.exit(main())
