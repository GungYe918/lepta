"""lepta DSL 파서 (MVP)
- %token / %ignore / %start / "lit":"lit" 선언
- %left / %right / %nonassoc / %use precedence NAME ;
- (신규) %recover off|panic ;
- (신규) %sync label(, label)* ;
- 규칙: RuleName : expr ;
- 시퀀스 말미: ... [%prec LABEL]?
- 세미콜론(;)은 모든 선언/규칙 종료에 **반드시 필요**
"""

from __future__ import annotations
import regex as re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .ast import *

# ---- Lexer 토큰 ----
_TOKEN_SPEC = [
    ("WS",       r"[ \t\f]+"),
    ("NEWLINE",  r"\n"),
    ("COMMENT",  r"//[^\n]*"),
    ("MCOMMENT", r"/\*.*?\*/"),
    ("PERCENT",  r"%"),
    ("COLON",    r":"),
    ("SEMI",     r";"),
    ("COMMA",    r","),          
    ("OR",       r"\|"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("QMARK",    r"\?"),
    ("STAR",     r"\*"),
    ("PLUS",     r"\+"),
    ("REGEX",    r"/(?:\\.|[^/])+/[imxsADU]*"),
    ("STRING",   r'"(?:\\.|[^"\\])*"'),
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),
]
MASTER_RE = re.compile("|".join(f"(?P<{n}>{p})" for n,p in _TOKEN_SPEC), re.S)

@dataclass
class Tok:
    kind: str
    lexeme: str
    start: int
    end: int
    line: int
    col: int

def _scan(src: str) -> List[Tok]:
    """개행은 줄/칼럼 갱신만 하고 토큰스트림에는 **넣지 않는다**."""
    toks: List[Tok] = []
    line = col = 1
    i = 0
    while i < len(src):
        m = MASTER_RE.match(src, i)
        if not m:
            raise SyntaxError(f"Unexpected char {src[i]!r} at {line}:{col}")
        kind = m.lastgroup or ""
        lex = m.group(0)
        start, end = i, m.end()
        nl_count = lex.count("\n")

        # NEWLINE도 WS/주석처럼 **토큰 미배출**
        if kind not in ("WS", "COMMENT", "MCOMMENT", "NEWLINE"):
            toks.append(Tok(kind, lex, start, end, line, col))

        # 위치 갱신
        if nl_count:
            line += nl_count
            col = len(lex) - lex.rfind("\n")
        else:
            col += len(lex)
        i = end

    toks.append(Tok("EOF", "", len(src), len(src), line, col))
    return toks


# ---------- error handling utils ----------
def _line_bounds(src: str, pos: int) -> Tuple[int, int]:
    """pos가 속한 라인의 [시작, 끝+1) 범위"""
    start = src.rfind("\n", 0, pos)
    if start == -1:
        start = 0
    else:
        start += 1
    end = src.find("\n", pos)
    if end == -1:
        end = len(src)
    return start, end

def _snippet_with_caret(src: str, tok: Tok) -> str:
    """토큰 시작 위치에 캐럿"""
    start, end = _line_bounds(src, tok.start)
    line_text = src[start:end]
    caret = " " * (tok.col - 1) + "^"
    return f"{line_text}\n{caret}"

def _snippet_caret_at_pos(src: str, pos: int) -> str:
    """임의의 절대 위치 pos에 캐럿(세미콜론 '바로 있어야 할 자리' 같은 곳)"""
    start, end = _line_bounds(src, pos)
    line_text = src[start:end]
    # 열(1-based) = pos - line_start + 1
    col = (pos - start) + 1
    caret = " " * (col - 1) + "^"
    return f"{line_text}\n{caret}"

# --- 토큰 스트림 ---
class _TS:
    def __init__(self, toks: List[Tok], src: str):
        self.toks = toks
        self.i = 0
        self.src = src

    def la(self) -> Tok:
        return self.toks[self.i]

    def eat(self, kind: str) -> Tok:
        t = self.la()
        if t.kind != kind:
            snippet = _snippet_with_caret(self.src, t)
            raise SyntaxError(
                f"Expected {kind}, got {t.kind} at {t.line}:{t.col}\n{snippet}"
            )
        self.i += 1
        return t

    def match(self, kind: str) -> Optional[Tok]:
        if self.la().kind == kind:
            return self.eat(kind)
        return None

def _unquote_string(s: str) -> str:
    return bytes(s[1:-1], "utf-8").decode("unicode_escape")

def _strip_regex(s: str) -> Tuple[str, str]:
    last = s.rfind("/")
    return s[1:last], s[last + 1:]

def _require_semi(ts: _TS, context: str, example: str, anchor: Optional[Tok] = None) -> None:
    """
    세미콜론 강제. 없으면:
      - Found: 다음 토큰/EOF 위치는 부가 정보로,
      - 캐럿은 anchor(직전 토큰)의 '끝 위치'에 찍음 → 올바른 줄에 표시됨.
    """
    if ts.match("SEMI"):
        return
    got = ts.la()
    where = f"{got.line}:{got.col}"
    found = "EOF" if got.kind == "EOF" else got.kind
    if anchor is not None:
        snippet = _snippet_caret_at_pos(ts.src, anchor.end)
    else:
        # 폴백: 다음 토큰 시작 위치
        snippet = _snippet_with_caret(ts.src, got)
    msg = (
        f"Missing ';' after {context} (semicolon is mandatory).\n"
        f"- Found: {found} at {where}\n"
        f"- Example: {example}\n\n"
        f"{snippet}"
    )
    raise SyntaxError(msg)

# --- 공용 파싱 유틸 ---
def _ident_or_string(ts: _TS) -> str:
    """IDENT 또는 STRING을 하나 소비해 일반 문자열로 돌려줍니다."""
    t = ts.la()
    if t.kind == "IDENT":
        return ts.eat("IDENT").lexeme
    if t.kind == "STRING":
        return _unquote_string(ts.eat("STRING").lexeme)
    raise SyntaxError(f"Expected IDENT or STRING, got {t.kind} at {t.line}:{t.col}")

def _parse_prec_decl(ts: _TS, g) -> None:
    """
    %left/%right/%nonassoc label... ;
    (주의: parse_grammar에서 '%'는 소비되었지만 assoc은 아직 소비되지 않은 상태일 때 호출)
    """
    assoc_tok = ts.eat("IDENT")  # 'left' | 'right' | 'nonassoc'
    assoc = assoc_tok.lexeme
    if assoc not in ("left", "right", "nonassoc"):
        raise SyntaxError(f"Unknown precedence assoc %{assoc} at {assoc_tok.line}:{assoc_tok.col}")
    labels: List[str] = []
    # 라벨 1개 이상
    while ts.la().kind in ("IDENT", "STRING"):
        labels.append(_ident_or_string(ts))
    # 세미콜론 강제
    _require_semi(ts, f"%{assoc} declaration", f"%{assoc} + - * / ;")
    g.decl_precedences.append(PrecedenceDecl(assoc=assoc, labels=labels))

def _parse_use_precedence(ts: _TS, g) -> None:
    """
    %use precedence NAME ;
    (주의: parse_grammar에서 '%'는 소비되었지만 'use'는 아직 소비되지 않은 상태일 때 호출)
    """
    kw = ts.eat("IDENT")  # 'use'
    if kw.lexeme != "use":
        raise SyntaxError(f"Unknown directive %{kw.lexeme}")
    key = ts.eat("IDENT")  # 'precedence'
    if key.lexeme != "precedence":
        raise SyntaxError(f"Expected 'precedence' after %use, got {key.lexeme}")
    name = ts.eat("IDENT").lexeme
    _require_semi(ts, "%use precedence", f"%use precedence {name};")
    g.decl_prec_templates.append(PrecedenceTemplateUse(name=name))

def _parse_recover_decl(ts: _TS, g: Grammar) -> None:
    """
    %recover off|panic ;
    """
    ident = ts.eat("IDENT")
    if ident.lexeme != "recover":
        raise SyntaxError(f"Unknown directive %{ident.lexeme}")
    mode_tok = ts.eat("IDENT")
    mode = mode_tok.lexeme
    if mode not in ("off", "panic"):
        snippet = _snippet_with_caret(ts.src, mode_tok)
        raise SyntaxError(f"Unknown recover mode '{mode}' at {mode_tok.line}:{mode_tok.col}\n{snippet}")
    _require_semi(ts, "%recover directive", "%recover panic;")
    g.recover_mode = mode

def _parse_sync_decl(ts: _TS, g: Grammar) -> None:
    """
    %sync LABEL(, LABEL)* ;
    LABEL은 IDENT, STRING, 또는 특별히 'EOF'를 허용.
    쉼표는 선택(공백만으로도 구분 가능)하지만, 예제/관습상 쉼표 권장.
    """
    ident = ts.eat("IDENT")
    if ident.lexeme != "sync":
        raise SyntaxError(f"Unknown directive %{ident.lexeme}")
    labels: List[str] = []
    # 최소 1개 필요
    first = True
    while True:
        # 라벨
        if ts.la().kind in ("IDENT", "STRING"):
            labels.append(_ident_or_string(ts))
            first = False
            # 선택적 쉼표
            if ts.match("COMMA"):
                continue
            # 다음 토큰이 또 라벨이면 공백 구분 허용
            elif ts.la().kind in ("IDENT", "STRING"):
                continue
            else:
                break
        else:
            if first:
                snippet = _snippet_with_caret(ts.src, ts.la())
                raise SyntaxError(f"%sync requires at least one label at {ts.la().line}:{ts.la().col}\n{snippet}")
            break
    _require_semi(ts, "%sync directive", '%sync ")", "]", "}", ";", EOF;')
    # 누적(중복 제거는 나중에)
    g.sync_labels.extend(labels)

# --- Grammar Parsing ---
def parse_grammar(src: str) -> Grammar:
    ts = _TS(_scan(src), src)
    g = Grammar()

    # 선언부: %token / %ignore / %start / %left|%right|%nonassoc / %use precedence / %recover / %sync / "lit":"lit" ;
    while True:
        t = ts.la()
        if t.kind == "PERCENT":
            ts.eat("PERCENT")
            # 공용: 여기서 어떤 지시어인지 분기
            look = ts.la()
            if look.kind != "IDENT":
                snippet = _snippet_with_caret(src, look)
                raise SyntaxError(f"Expected directive name after '%', got {look.kind} at {look.line}:{look.col}\n{snippet}")

            # 토큰을 들여다보고 케이스 분기
            if look.lexeme in ("token", "ignore", "start"):
                ident_tok = ts.eat("IDENT")
                ident = ident_tok.lexeme
                if ident == "token":
                    name = ts.eat("IDENT").lexeme
                    regex_tok = ts.eat("REGEX")
                    pat, flags = _strip_regex(regex_tok.lexeme)
                    g.decl_tokens.append(TokenDecl(name, pat, flags))
                    _require_semi(ts, "%token declaration", '%token NAME /regex/;', anchor=regex_tok)
                elif ident == "ignore":
                    regex_tok = ts.eat("REGEX")
                    pat, flags = _strip_regex(regex_tok.lexeme)
                    g.decl_ignores.append(IgnoreDecl(pat, flags))
                    _require_semi(ts, "%ignore declaration", r'%ignore /\s+/;', anchor=regex_tok)
                elif ident == "start":
                    start_tok = ts.eat("IDENT")
                    g.start = start_tok.lexeme
                    _require_semi(ts, "%start declaration", '%start StartSymbol;', anchor=start_tok)

            elif look.lexeme in ("left", "right", "nonassoc"):
                # %left/%right/%nonassoc label... ;
                _parse_prec_decl(ts, g)

            elif look.lexeme == "use":
                # %use precedence NAME ;
                _parse_use_precedence(ts, g)

            elif look.lexeme == "recover":
                _parse_recover_decl(ts, g)

            elif look.lexeme == "sync":
                _parse_sync_decl(ts, g)

            else:
                ident_tok = ts.eat("IDENT")  # 소모해서 에러 위치를 정확히
                snippet = _snippet_with_caret(src, ident_tok)
                raise SyntaxError(
                    f"Unknown directive %{ident_tok.lexeme} at {ident_tok.line}:{ident_tok.col}\n{snippet}"
                )

        elif t.kind == "STRING":
            # 키워드 매핑: "lit" : "lit" ;
            lit1_tok = ts.eat("STRING")
            lit1 = _unquote_string(lit1_tok.lexeme)
            ts.eat("COLON")
            lit2_tok = ts.eat("STRING")
            lit2 = _unquote_string(lit2_tok.lexeme)
            if lit1 != lit2:
                snippet = _snippet_with_caret(src, lit1_tok)
                raise SyntaxError(
                    f'Keyword mapping must be identical on both sides: "{lit1}" : "{lit2}"\n{snippet}'
                )
            g.decl_keywords.append(KeywordDecl(lit1))
            _require_semi(ts, 'keyword literal mapping (e.g. "+" : "+")', '"+" : "+";', anchor=lit2_tok)
        else:
            break

    # 규칙부: RuleName : expr ;
    while ts.la().kind != "EOF":
        lhs_tok = ts.eat("IDENT")
        lhs = lhs_tok.lexeme
        ts.eat("COLON")
        expr = _parse_expr(ts)
        # expr 마지막으로 소비된 토큰을 anchor로 사용
        last_tok = ts.toks[ts.i - 1] if ts.i > 0 else lhs_tok
        _require_semi(ts, f"rule '{lhs}'", f"{lhs} : ... ;", anchor=last_tok)
        g.rules.append(Rule(lhs, expr))

    if not g.start and g.rules:
        g.start = g.rules[0].name

    # %sync 중복 제거(보수적으로 보존 순서 유지)
    if g.sync_labels:
        seen = set()
        uniq = []
        for s in g.sync_labels:
            if s not in seen:
                uniq.append(s); seen.add(s)
        g.sync_labels = uniq

    # recover_mode 기본값(off) 처리(명시 None이면 off 취급)
    if not g.recover_mode:
        g.recover_mode = "off"

    return g

def _parse_expr(ts: _TS) -> Expr:
    alts = [_parse_seq(ts)]
    while ts.match("OR"):
        alts.append(_parse_seq(ts))
    return Expr(alts)

def _parse_seq(ts: _TS) -> Seq:
    """
    시퀀스: (IDENT | STRING | "(" expr ")")+ ["%prec" LABEL]?
    """
    items: List[Atom] = []
    while True:
        if ts.la().kind in ("IDENT", "STRING", "LPAREN"):
            items.append(_parse_atom(ts))
        else:
            break

    # 선택적 %prec
    prec_label: Optional[str] = None
    if ts.la().kind == "PERCENT":
        # 후보: %prec
        ts.eat("PERCENT")
        ident = ts.eat("IDENT").lexeme
        if ident != "prec":
            raise SyntaxError(f"Unknown %directive %{ident} inside a rule; did you mean '%prec'?")
        prec_label = _ident_or_string(ts)

    return Seq(items, prec=prec_label)

def _parse_atom(ts: _TS) -> Atom:
    t = ts.la()
    if t.kind == "IDENT":
        node = Name(ts.eat("IDENT").lexeme)
    elif t.kind == "STRING":
        node = Lit(_unquote_string(ts.eat("STRING").lexeme))
    elif t.kind == "LPAREN":
        ts.eat("LPAREN")
        node = Group(_parse_expr(ts))
        ts.eat("RPAREN")
    else:
        snippet = _snippet_with_caret(ts.src, t)
        raise SyntaxError(f"Unexpected token {t.kind} at {t.line}:{t.col}\n{snippet}")
    suf = Suffix.NONE
    if ts.match("QMARK"):
        suf = Suffix.OPT
    elif ts.match("STAR"):
        suf = Suffix.STAR
    elif ts.match("PLUS"):
        suf = Suffix.PLUS
    return Atom(node, suf)
