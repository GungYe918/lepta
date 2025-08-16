# lepta/lex/__init__.py
"""lepta 토크나이저(runtime) — Grammar 선언으로부터 동작하는 간단 렉서.

특징
----
- 문법(AST) 선언부(%token, %ignore, "키워드")를 바탕으로 토큰 스트림을 생성
- **키워드 리터럴 우선** 매칭 → 이후 %token 정규식 매칭
- %ignore 패턴은 입력에서 스킵
- 산출 토큰의 `type`은 lepta 내부 심볼 이름과 **정확히 일치**
  - 키워드: 해당 리터럴 그대로 (예: "+", "(", ")")
  - %token: 토큰 이름 그대로 (예: "NUMBER")


매칭 순서:
 1) %ignore 패턴을 가능한 만큼 스킵
  2) 키워드(리터럴) — **길이 내림차순(최장일치)**, 알파벳/유니코드 키워드는 단어경계 검사
  3) PEG 토큰 — **최장일치**, 동률이면 선언 순서
     - [trigger='x'] 지정 시 해당 문자로 시작할 때만 PEG 매칭 시도
  4) %token 정규식 — **가장 긴 매치**(동률이면 선언 순서)
  5) 둘 다 불일치 → SyntaxError

  
API
---
- `LexTok(type: str, text: str, line: int, col: int)` — 토큰 단위
- `Lexer` 프로토콜
    - `peek() -> Optional[LexTok]`
    - `next() -> Optional[LexTok]`
    - `next_kind(sym: SymbolTable) -> Optional[int]`  # 파서용 단말 ID
- `SimpleLexer.from_grammar(g: Grammar)` — Grammar에서 직접 렉서 만들기
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Pattern, Dict, TYPE_CHECKING
import re


# --- PEG 런타임 의존(프로젝트 내부 모듈) ---
try:
    # peg 모듈에서 노출된 공용 API를 사용합니다.
    # 예상 인터페이스:
    #   PegProgram.from_source(src: str, name: str|None=None) -> PegProgram
    #   PegRunner(prog: PegProgram).run(rule: str, text: str, pos: int) -> tuple[bool, int]
    from ..peg import PegRunner  # type: ignore
    _HAS_PEG = True
except Exception:
    PegRunner = object   # type: ignore
    _HAS_PEG = False

if TYPE_CHECKING:
    from ..peg import PegProgram as _PegProgram
else:
    class _PegProgram:  # 런타임 더미
        pass


try:
    import regex as _uregex
    _HAS_UREGEX = True
    _RE_XID_CONT = _uregex.compile(r"\p{XID_Continue}")
except Exception:
    _HAS_UREGEX = False
    _RE_XID_CONT = None

def _is_ident_continue(ch: str) -> bool:
    """유니코드 식별자 이어붙임 문자(XID_Continue) 근사 판정.
    - `regex`가 있으면 \p{XID_Continue}로 엄밀 판정
    - 없으면 fallback: ch.isalnum() or ch == '_'
    """
    if _HAS_UREGEX:  # type: ignore[truthy-bool]
        # fullmatch가 가장 정확(단일 코드포인트)
        return bool(_RE_XID_CONT.fullmatch(ch))  # type: ignore[union-attr]
    return ch.isalnum() or ch == "_"

def _is_word_keyword(s: str) -> bool:
    """키워드가 '단어' 성격(식별자 문자 포함)을 가지면 True.
    유니코드 식별자 경계로 판정한다.
    """
    return any(_is_ident_continue(c) for c in s)

# --------- Public datatypes ---------

@dataclass(frozen=True)
class LexTok:
    type: str   # 내부 심볼 이름(키워드 리터럴 또는 %token 이름)
    text: str   # 원문 lexeme
    line: int   # 1-based
    col: int    # 1-based

class Lexer:
    """런타임이 기대하는 최소 인터페이스."""
    def peek(self) -> Optional[LexTok]:
        raise NotImplementedError
    def next(self) -> Optional[LexTok]:
        raise NotImplementedError
    def next_kind(self, sym) -> Optional[int]:
        """심볼테이블에 맞춘 **단말 ID**를 돌려준다.
        - 토큰이 없으면 None(=EOF로 간주).
        """
        t = self.next()
        if t is None:
            return None
        try:
            return sym.id_of(t.type)
        except Exception:
            raise KeyError(f"Unknown terminal name for token: {t.type!r}")

# --------- Helpers ---------

_FLAG_MAP = {
    'i': re.IGNORECASE,
    'm': re.MULTILINE,
    's': re.DOTALL,
    'x': re.VERBOSE,
    'A': re.ASCII,
    # 'U'는 Py3 기본 유니코드라 무시
}

def _compile_regex(pat: str, flags: str) -> Pattern[str]:
    f = 0
    for ch in flags:
        f |= _FLAG_MAP.get(ch, 0)
    return re.compile(pat, f)

# --------- PEG 토큰 룰 ---------

@dataclass(frozen=True)
class _PegTokenRule:
    term: str
    block: str
    rule: str
    trigger: Optional[str]  # 첫 글자 트리거(선택)

# --------- Core implementation ---------

class SimpleLexer(Lexer):
    """
    SimpleLexer
    ===========
    Grammar의 선언부를 사용해 작동하는 참조 구현.
    (키워드 => PEG => 정규식)

    매칭 순서:
      1) %ignore 패턴을 가능한 만큼 스킵
      2) 키워드(리터럴) — **선언 순서대로**, 알파벳 키워드는 단어경계 검사
      3) %token 정규식 — **가장 긴 매치**를 선택(같으면 먼저 선언된 것)
      4) 둘 다 불일치 → SyntaxError
    """
    def __init__(self,
            keywords: List[str],
            tokens: List[Tuple[str, Pattern[str]]],
            ignores: List[Pattern[str]],
            peg_programs: Optional[Dict[str, "_PegProgram"]] = None,
            peg_rules: Optional[List["_PegTokenRule"]] = None):
            self._keywords = keywords[:]         # 리터럴 문자열
            self._tokens = tokens[:]             # (name, compiled_regex)
            self._ignores = ignores[:]           # compiled_regex
            self._peg_programs = dict(peg_programs or {})
            self._peg_rules = list(peg_rules or [])
            self._text = ""
            self._i = 0
            self._line = 1
            self._col = 1
            self._peek_cache: Optional[LexTok] = None

    # ---- Constructors ----
    @classmethod
    def from_grammar(cls, g) -> "SimpleLexer":
        """Grammar(AST)의 선언부로부터 렉서를 생성."""
        # 1) 키워드: 중복 제거(선언 첫 등장만 유지) + 길이 내림차순(동길이면 선언 순서)
        raw_kws = [kw.lexeme for kw in getattr(g, "decl_keywords", [])]
        seen = set()
        kws_pairs = []
        for idx, lit in enumerate(raw_kws):
            if lit in seen:
                continue
            seen.add(lit)
            kws_pairs.append((lit, idx))
        kws_pairs.sort(key=lambda p: (-len(p[0]), p[1]))
        keywords = [lit for (lit, _idx) in kws_pairs]

        # 2) %token 정규식
        tokens: List[Tuple[str, Pattern[str]]] = []
        for td in getattr(g, "decl_tokens", []):
            tokens.append((td.name, _compile_regex(td.pattern, td.flags or "")))

        # 3) %ignore
        ignores = [_compile_regex(ig.pattern, ig.flags or "") for ig in getattr(g, "decl_ignores", [])]

        # 4) PEG 프로그램/룰
        peg_programs: Dict[str, "_PegProgram"] = {}
        peg_rules: List[_PegTokenRule] = []
        if _HAS_PEG:
            # 지역 import로 런타임 의존성을 최소화
            from ..peg import PegProgram  # type: ignore
            # %peg 블록 컴파일
            for pb in getattr(g, "decl_peg_blocks", []):
                try:
                    prog = PegProgram.from_source(pb.src)
                except Exception as e:
                    raise SyntaxError(f"PEG block '{pb.name}' compile error: {e}")
                peg_programs[pb.name] = prog  # block name -> program
            # %token ... %peg(Block.Rule) [trigger='x']
            for pt in getattr(g, "decl_peg_tokens", []):
                block, rule = pt.peg_ref
                trig = pt.trigger if pt.trigger else None
                peg_rules.append(_PegTokenRule(term=pt.name, block=block, rule=rule, trigger=trig))
        else:
            # PEG 런타임이 없는데 PEG 토큰이 선언되면 바로 실패시켜 안내
            if getattr(g, "decl_peg_tokens", []):
                raise RuntimeError("PEG runtime is not available but PEG tokens were declared.")

        return cls(keywords, tokens, ignores, peg_programs, peg_rules)

    
    # ---- Input binding ----
    def reset(self, text: str, *, line: int = 1, col: int = 1) -> None:
        self._text = text
        self._i = 0
        self._line = line
        self._col = col
        self._peek_cache = None

    # ---- Public API ----
    def peek(self) -> Optional[LexTok]:
        if self._peek_cache is None:
            self._peek_cache = self._next_token()
        return self._peek_cache

    def next(self) -> Optional[LexTok]:
        if self._peek_cache is not None:
            t = self._peek_cache
            self._peek_cache = None
            return t
        return self._next_token()

    # ---- Internals ----
    def _advance_text(self, consumed: str) -> None:
        """소비된 텍스트 길이만큼 내부 포인터/행렬을 갱신."""
        n = len(consumed)
        seg = consumed
        while True:
            j = seg.find("\n")
            if j == -1:
                break
            self._line += 1
            self._col = 1
            seg = seg[j+1:]
        self._col += len(seg)
        self._i += n

    def _skip_ignores(self) -> None:
        while self._i < len(self._text):
            progressed = False
            rest = self._text[self._i:]
            for rgx in self._ignores:
                m = rgx.match(rest)
                if m and m.end() > 0:
                    self._advance_text(m.group(0))
                    progressed = True
                    break
            if not progressed:
                return

    def _match_keyword(self) -> Optional[LexTok]:
        s = self._text
        i = self._i
        for lit in self._keywords:  # 이미 길이 내림차순으로 정렬됨
            if not s.startswith(lit, i):
                continue
            # 유니코드 '단어 키워드'는 경계 검사(앞/뒤).
            if lit and _is_word_keyword(lit):
                # prev boundary
                prev_ok = True
                if i > 0:
                    prev_ch = s[i-1]
                    if _is_ident_continue(prev_ch):
                        prev_ok = False
                # next boundary
                j = i + len(lit)
                next_ok = True
                if j < len(s):
                    next_ch = s[j]
                    if _is_ident_continue(next_ch):
                        next_ok = False
                if not (prev_ok and next_ok):
                    continue
            return LexTok(type=lit, text=lit, line=self._line, col=self._col)
        return None
    
    def _peg_try_match(self) -> Optional[LexTok]:
        """선언부 %token … %peg(Block.Rule) 항목을 트리거 기반으로 국소 매칭."""
        if not self._peg_rules:
            return None
        text = self._text
        i = self._i
        best_len = -1
        best_term: Optional[str] = None

        for r in self._peg_rules:
            # 트리거가 있으면 첫 글자 일치할 때만 시도(광범위 백트래킹 방지)
            if r.trigger is not None:
                if i >= len(text) or text[i] != r.trigger:
                    continue
            prog = self._peg_programs.get(r.block)
            if prog is None:
                continue

            try:
                ok, end = PegRunner(prog).run(r.rule, text, i)  # ← 실제 시그니처
            except Exception:
                ok, end = (False, i)

            if ok and end > i:
                l = end - i
                if l > best_len:
                    best_len = l
                    best_term = r.term

        if best_len > 0 and best_term is not None:
            lexeme = text[i:i+best_len]
            return LexTok(type=best_term, text=lexeme, line=self._line, col=self._col)
        return None


    def _match_token_regex(self) -> Optional[LexTok]:
        rest = self._text[self._i:]
        best_name = None
        best_text = None
        best_len = -1
        for name, rgx in self._tokens:
            m = rgx.match(rest)
            if not m:
                continue
            txt = m.group(0)
            if len(txt) > best_len:
                best_len = len(txt)
                best_name = name
                best_text = txt
        if best_len > 0:
            return LexTok(type=best_name, text=best_text, line=self._line, col=self._col)
        return None

    def _next_token(self) -> Optional[LexTok]:
        if self._i >= len(self._text):
            return None
        self._skip_ignores()
        if self._i >= len(self._text):
            return None

        # 1) 키워드
        kw = self._match_keyword()
        if kw is not None:
            self._advance_text(kw.text)
            return kw

        # 2) PEG 토큰(국소, 트리거 기반, 최장일치)
        peg = self._peg_try_match()
        if peg is not None:
            self._advance_text(peg.text)
            return peg

        # 3) 정규식 토큰(최장일치)
        tk = self._match_token_regex()
        if tk is not None:
            self._advance_text(tk.text)
            return tk

        # 실패
        ch = self._text[self._i]
        raise SyntaxError(f"Lexing error: unexpected character {ch!r} at {self._line}:{self._col}")


# Convenience
def build_lexer_from_grammar(g) -> SimpleLexer:
    lx = SimpleLexer.from_grammar(g)
    return lx
