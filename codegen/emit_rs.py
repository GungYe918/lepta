# lepta/codegen/emit_rs.py
"""Rust Code Emit (단일 .rs 파일 생성; 런타임 포함).

개요
----
- CodegenIR을 받아 Rust 소스 코드를 **문자열로** 생성한다.
- 방출되는 Rust 파일은:
  * 테이블 상수(ACTION/GOTO/PROD 등)
  * 고속 파서 런타임(스택 머신)
  * 최소한의 API (Lexer 트레이트 + parse 함수)
  * (+옵션) 간단 토크나이저(SimpleLexer)         ← lexer_ast 인자 전달 시 포함
  * 액션 기반 값 생성 런타임(parse_with_actions)
  * (+옵션) 오류 복구(PANIC 모드): %recover / %sync 지시어 사용 시 활성화

성능 개선
---------
- (구) MVP 런타임은 상태별 ACTION/GOTO를 **선형 탐색**으로 찾았다.
- (신) 본 버전은 ACTION/GOTO를 **Dense 2D 테이블(행렬)** 로 방출하여
  **상수시간 O(1)** 조회가 가능하다.

  * ACTION_KIND: i8[S * T]  (0=None, 1=shift, 2=reduce, 3=accept)
  * ACTION_ARG : i32[S * T] (shift의 next_state 또는 reduce의 prod_idx)
  * GOTO_NEXT  : i32[S * N] (-1=None, 그 외 next_state)

  where S=n_states, T=n_terms, N=n_nonterms.

메모리 주의
-----------
- Dense 테이블은 (S*T) / (S*N) 크기를 가진다.
- 문법이 매우 크고 희소한 경우 메모리가 증가할 수 있으니, 필요시
  스파스 포맷(기존 선형 탐색)으로 되돌리는 옵션을 추후 추가할 수 있다.
"""

from __future__ import annotations
from typing import List, Optional
from .ir import CodegenIR, SHIFT, REDUCE, ACCEPT


# ---------- 유틸 ----------

def _fmt_list_int(ints: List[int]) -> str:
    """정수 리스트를 Rust 소스 상수 배열로 직렬화한다."""
    return ", ".join(str(x) for x in ints)


def _preflight_check(ir: CodegenIR) -> None:
    """기본 불변식/전제 조건을 조기 검증하여 생성 단계에서 실패시킨다."""
    if ir.n_states <= 0 or ir.n_terms <= 0 or ir.n_nonterms <= 0:
        raise ValueError(f"emit_rs: invalid sizes (states={ir.n_states}, terms={ir.n_terms}, nonterms={ir.n_nonterms})")
    if len(ir.terms) != ir.n_terms:
        raise ValueError(f"emit_rs: len(terms)={len(ir.terms)} != n_terms={ir.n_terms}")
    if ir.terms[0] != "$":
        raise ValueError("emit_rs: TERM_NAMES[0] must be '$' (EOF)")
    if len(ir.nonterms) != ir.n_nonterms:
        raise ValueError(f"emit_rs: len(nonterms)={len(ir.nonterms)} != n_nonterms={ir.n_nonterms}")
    if len(ir.prod_lhs) != len(ir.prod_rhs_len):
        raise ValueError("emit_rs: PROD_LHS/PROD_RHS_LEN length mismatch")


def _normalize_prod_lhs_to_local(ir: CodegenIR) -> List[int]:
    """
    PROD_LHS를 '로컬 비단말 인덱스(0..N_NONTERMS-1)'로 정규화하여 반환한다.
    스키마 판정 규칙:
      - 엄격 글로벌 값( N_TERMS < v < N_TERMS+N_NONTERMS )이 하나라도 있으면 '글로벌 스키마'
        → 모든 v >= N_TERMS 값을 (v - N_TERMS)로 변환
      - 그렇지 않으면 '로컬 스키마'
        → 모든 v는 0..N_NONTERMS-1 범위여야 함
    경계값 v == N_TERMS 는 글로벌 스키마일 때 글로벌로 해석(=0), 로컬 스키마일 때는 로컬로 유지.
    """
    NT = ir.n_terms
    NNT = ir.n_nonterms
    vals = ir.prod_lhs

    # 유효성 1차 체크
    for i, v in enumerate(vals):
        if not (0 <= v < NNT or NT <= v < NT + NNT):
            raise ValueError(
                f"emit_rs: PROD_LHS[{i}] invalid lhs={v} "
                f"(expect local [0..{NNT-1}] or global [{NT}..{NT+NNT-1}])"
            )

    # 스키마 결정
    has_strict_global = any(NT < v < NT + NNT for v in vals)
    use_global = has_strict_global

    out: List[int] = []
    if use_global:
        for i, v in enumerate(vals):
            if v >= NT:
                out.append(v - NT)
            else:
                raise ValueError(
                    f"emit_rs: PROD_LHS uses mixed schemes (found local {v} with global scheme); index={i}"
                )
    else:
        for i, v in enumerate(vals):
            if 0 <= v < NNT:
                out.append(v)
            else:
                raise ValueError(
                    f"emit_rs: PROD_LHS uses mixed schemes (found global-like {v} with local scheme); index={i}"
                )

    assert all(0 <= v < NNT for v in out), "emit_rs: normalized PROD_LHS out of range"
    return out


def _fmt_action_dense(ir: CodegenIR) -> str:
    """
    ACTION을 Dense 2D 테이블로 직렬화한다.
    - KIND: 0=None, 1=shift, 2=reduce, 3=accept
    - ARG : shift의 다음 상태 또는 reduce의 프로덕션 인덱스
    """
    S = ir.n_states
    T = ir.n_terms
    kind = [0] * (S * T)     # i8
    arg  = [0] * (S * T)     # i32

    for s, row in enumerate(ir.action_rows):
        base = s * T
        for (term, k, a) in row:
            if not (0 <= term < T):
                raise ValueError(f"emit_rs: ACTION term out of range: term={term}, n_terms={T}, state={s}")
            if k not in (SHIFT, REDUCE, ACCEPT):
                raise ValueError(f"emit_rs: ACTION kind invalid: kind={k}, expected in {{SHIFT({SHIFT}), REDUCE({REDUCE}), ACCEPT({ACCEPT})}}")
            idx = base + term
            kind[idx] = int(k)
            arg[idx]  = int(a)

    if len(kind) != S * T or len(arg) != S * T:
        raise AssertionError("emit_rs: ACTION dense table size mismatch")

    kind_src = ", ".join(str(x) for x in kind)
    arg_src  = ", ".join(str(x) for x in arg)
    return f"""\
/// Dense ACTION 테이블 (O(1) 조회).
/// 인덱스 계산: `idx = state * N_TERMS + term`
/// - `ACTION_KIND[idx]` == 0: 액션 없음(에러)
/// - `ACTION_KIND[idx]` == 1: shift  → `ACTION_ARG[idx]`가 다음 상태
/// - `ACTION_KIND[idx]` == 2: reduce → `ACTION_ARG[idx]`가 프로덕션 인덱스
/// - `ACTION_KIND[idx]` == 3: accept(수용)
///
/// 참고: term은 `Lexer::next_kind()`가 돌려주는 **단말 ID(u16)** 이어야 하며,
/// 해당 ID의 문자열 이름은 `TERM_NAMES[term as usize]`로 확인할 수 있다.
pub static ACTION_KIND: &[i8] = &[{kind_src}];
pub static ACTION_ARG:  &[i32] = &[{arg_src}];
"""


def _fmt_goto_dense(ir: CodegenIR) -> str:
    """
    GOTO를 Dense 2D 테이블로 직렬화한다.
    - 값: next_state (없으면 -1)
    - aid(비단말 ID)는 입력 스키마(로컬/글로벌)를 먼저 판정한 뒤 일괄 정규화한다.
    """
    S = ir.n_states
    N = ir.n_nonterms
    NT = ir.n_terms

    nexts = [-1] * (S * N)

    all_aids = [aid for row in ir.goto_rows for (aid, _ns) in row]
    if not all(0 <= a < N or NT <= a < NT + N for a in all_aids):
        bad = [a for a in all_aids if not (0 <= a < N or NT <= a < NT + N)]
        raise ValueError(f"emit_rs: GOTO nonterm id out of range: {bad[:5]}...")

    has_strict_global = any(NT < a < NT + N for a in all_aids)
    use_global = has_strict_global

    for s, row in enumerate(ir.goto_rows):
        base = s * N
        for (aid, ns) in row:
            if use_global:
                if aid >= NT:
                    a_local = aid - NT
                else:
                    raise ValueError(
                        f"emit_rs: GOTO mixed schemes (found local aid={aid} with global scheme); state={s}"
                    )
            else:
                if 0 <= aid < N:
                    a_local = aid
                else:
                    raise ValueError(
                        f"emit_rs: GOTO mixed schemes (found global-like aid={aid} with local scheme); state={s}"
                    )

            if not (0 <= ns < S):
                raise ValueError(
                    f"emit_rs: GOTO next state out of range: ns={ns}, n_states={S}, state={s}, A_local={a_local}"
                )

            idx = base + a_local
            nexts[idx] = int(ns)

    if len(nexts) != S * N:
        raise AssertionError("emit_rs: GOTO dense table size mismatch")

    nxt_src = ", ".join(str(x) for x in nexts)
    return f"""\
/// Dense GOTO 테이블 (O(1) 조회).
/// 인덱스 계산: `idx = state * N_NONTERMS + nonterm`
/// - `GOTO_NEXT[idx]` < 0 : 이동 없음(잘못된 축약/테이블 생성 오류)
/// - `GOTO_NEXT[idx]` >= 0: 다음 상태 번호
///
/// reduce 후, 스택 top 상태를 t라고 할 때:
/// `goto_state(t, PROD_LHS[p])`가 새로운 상태를 제공한다.
pub static GOTO_NEXT: &[i32] = &[{nxt_src}];
"""


def _escape_rs(s: str) -> str:
    """Rust 문자열 리터럴 이스케이프."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _lexer_src_from_ast(ast) -> str:
    """
    간단 러스트 렉서 방출:
    - 키워드(리터럴) 우선, 유니코드 경계( is_alphanumeric() || '_' )로 단어 키워드 구분
    - %ignore: /\s+/ 이 있으면 공백/개행 스킵
    - %token: NUMBER만 내장 스캔(그 외는 렉서 직접 구현 권장)
    """
    kws = sorted({kw.lexeme for kw in ast.decl_keywords}, key=lambda s: (-len(s), s))
    kws_arr = ", ".join(f"\"{_escape_rs(k)}\"" for k in kws)
    has_ws_ignore = any(ig.pattern == r"\s+" for ig in ast.decl_ignores)
    token_names = [td.name for td in ast.decl_tokens]
    has_number = "NUMBER" in token_names

    if has_ws_ignore:
        skip_ignores_code = (
            "while let Some(ch) = self.peek_char() { "
            "if ch.is_whitespace() { "
            "self.take_while(|c| c.is_whitespace()); "
            "} else { break; } }"
        )
    else:
        skip_ignores_code = "/* no %ignore /\\s+/; */"

    if has_number:
        match_number_code = """
        // optional sign
        if let Some('-') = self.peek_char() { self.advance_by(1); }
        // int part
        if self.starts_with_at("0") {
            self.advance_by(1);
        } else {
            if let Some(ch) = self.peek_char() {
                if ch >= '1' && ch <= '9' {
                    self.advance_by(1);
                    self.take_while(|c| c.is_ascii_digit() || c == '_');
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        // frac
        if self.starts_with_at(".") {
            self.advance_by(1);
            let n = self.take_while(|c| c.is_ascii_digit() || c == '_');
            if n == 0 { return None; }
        }
        // exp
        if let Some(ch) = self.peek_char() {
            if ch == 'e' || ch == 'E' {
                self.advance_by(1);
                if let Some(sign) = self.peek_char() {
                    if sign == '+' || sign == '-' { self.advance_by(1); }
                }
                let n = self.take_while(|c| c.is_ascii_digit() || c == '_');
                if n == 0 { return None; }
            }
        }
        Some("NUMBER")
        """.strip("\n")
    else:
        match_number_code = "return None;"

    # NEW: 유니코드 경계 판정 헬퍼 및 단어 키워드 판정
    return f"""
// ====== Generated Simple Lexer (no external crates) ======
// 유니코드 경계: 식별자 문자는 `ch.is_alphanumeric() || ch == '_'`로 근사 처리합니다.
// 제한: 공백 스킵(\\s+), 키워드 리터럴, NUMBER 토큰만 지원합니다.

pub struct SimpleLexer<'a> {{
    text: &'a str,
    i: usize,
}}

impl<'a> SimpleLexer<'a> {{
    pub fn new(input: &'a str) -> Self {{
        Self {{ text: input, i: 0 }}
    }}

    #[inline] fn at_eof(&self) -> bool {{ self.i >= self.text.len() }}
    #[inline] fn peek_char(&self) -> Option<char> {{ self.text[self.i..].chars().next() }}
    #[inline] fn advance_by(&mut self, n_bytes: usize) {{ self.i += n_bytes; }}
    #[inline] fn starts_with_at(&self, s: &str) -> bool {{ self.text[self.i..].as_bytes().starts_with(s.as_bytes()) }}

    #[inline]
    fn take_while<F: FnMut(char)->bool>(&mut self, mut f: F) -> usize {{
        let mut n = 0usize;
        for ch in self.text[self.i..].chars() {{
            if f(ch) {{ n += ch.len_utf8(); }} else {{ break; }}
        }}
        self.advance_by(n);
        n
    }}

    #[inline]
    fn is_ident_continue(ch: char) -> bool {{
        ch.is_alphanumeric() || ch == '_'
    }}

    #[inline]
    fn is_word_kw(s: &str) -> bool {{
        s.chars().any(|c| Self::is_ident_continue(c))
    }}

    fn match_keyword(&mut self) -> Option<&'static str> {{
        const KEYWORDS: &[&str] = &[{kws_arr}];
        for &kw in KEYWORDS {{
            if self.starts_with_at(kw) {{
                // '단어 키워드'면 다음 문자가 식별자면 안 됨(유니코드 경계)
                if Self::is_word_kw(kw) {{
                    let after = self.i + kw.len();
                    if after < self.text.len() {{
                        if let Some(next) = self.text[after..].chars().next() {{
                            if Self::is_ident_continue(next) {{ continue; }}
                        }}
                    }}
                }}
                self.advance_by(kw.len());
                return Some(kw);
            }}
        }}
        None
    }}

    fn skip_ignores(&mut self) {{
        {skip_ignores_code}
    }}

    fn match_number(&mut self) -> Option<&'static str> {{
        {match_number_code}
    }}
}}

impl<'a> Lexer for SimpleLexer<'a> {{
    fn next_kind(&mut self) -> Option<u16> {{
        self.skip_ignores();
        if self.at_eof() {{ return Some(EOF_TERM); }}

        if let Some(kw) = self.match_keyword() {{
            // TERM_NAMES에서 id 찾기
            for (i, name) in TERM_NAMES.iter().enumerate() {{
                if *name == kw {{ return Some(i as u16); }}
            }}
            panic!("keyword not found in TERM_NAMES: {{}}", kw);
        }}

        if let Some(tok) = self.match_number() {{
            for (i, name) in TERM_NAMES.iter().enumerate() {{
                if *name == tok {{ return Some(i as u16); }}
            }}
            panic!("token not found in TERM_NAMES: {{}}", tok);
        }}

        let bad = self.text[self.i..].chars().next().unwrap();
        panic!("Lexing error: unsupported token starting with '{{:?}}' at byte {{}}", bad, self.i);
    }}
}}
""".lstrip("\n")



# ---------- 메인 방출기 ----------

def emit_rs_to_string(ir: CodegenIR, module_name: str = "parser", *, lexer_ast: Optional[object] = None) -> str:
    """
    emit_rs_to_string(ir, module_name, lexer_ast=None) -> str
    --------------------------------------------------------
    CodegenIR을 받아 **하나의 Rust 소스 문자열**을 생성한다.
    lexer_ast 가 주어지면, 간단 러스트 렉서(SimpleLexer)도 함께 방출한다.
    """
    _preflight_check(ir)
    lhs_local = _normalize_prod_lhs_to_local(ir)

    # 오류 복구 상수 생성
    recover_panic = (ir.recover_mode or "off") == "panic" and len(ir.sync_term_ids) > 0
    sync_terms_src = ", ".join(str(t) for t in (ir.sync_term_ids or []))
    max_err = int(ir.max_errors or 0)

    header = f"""\
//! Auto-generated by lepta
//! module: {module_name}
//! states={ir.n_states}, terms={ir.n_terms}, nonterms={ir.n_nonterms}
//!
//! 이 파일은 lepta가 `.g` 문법에서 생성한 **LALR/SLR 파서**의
//! **런타임(스택 머신) + 테이블**을 한 파일에 담은 것이다.
//! - 사용 방법: `mod parser; use parser::{{parse, Lexer}}`
//! - 필요 요소: `Lexer` 트레이트 구현(다음 토큰의 **종류 ID**를 돌려줌)
//! - 파싱 결과: 성공 시 `Ok(())`, 실패 시 `ParseError`
//!
//! ## 테이블 레이아웃(고성능 O(1) 조회)
//! 본 런타임은 **Dense 2D 테이블**을 사용하여 ACTION/GOTO를 O(1)에 조회
//! - `ACTION_KIND[s*T + t]`: 상태 `s`에서 단말 `t`일 때의 동작(0=None, 1=shift, 2=reduce, 3=accept)
//! - `ACTION_ARG [s*T + t]`: 동작 인자(shift=다음 상태, reduce=프로덕션 인덱스)
//! - `GOTO_NEXT [s*N + A]`: 상태 `s`에서 비단말 `A`로의 다음 상태(없으면 -1)
//!
//! ## 오류 복구(PANIC 모드)
//! - %recover panic; 과 %sync ...; 를 지정하면 활성화됩니다.
//! - 에러 시 입력을 동기화 토큰까지 스킵하고, 스택을 팝하며 재동기화합니다.
//!
//! 메모리 사용량은 S×T, S×N 크기에 비례한다.
#![allow(non_snake_case)]
#![allow(dead_code)]
"""

    # 메타 + 프로덕션
    prod_lhs_src = _fmt_list_int(lhs_local)
    prod_rhs_len_src = _fmt_list_int(ir.prod_rhs_len)
    prods_src = f"""\
/// 파서 테이블 메타데이터
pub const N_STATES: usize = {ir.n_states};
pub const N_TERMS:  usize = {ir.n_terms};
pub const N_NONTERMS: usize = {ir.n_nonterms};
pub const START_STATE: usize = {ir.start_state};
pub const EOF_TERM: u16 = 0; // terms[0] == "$"

/// 오류 복구 메타
pub const RECOVER_PANIC: bool = {str(recover_panic).lower()};
pub const MAX_ERRORS: usize = {max_err};
pub static SYNC_TERMS: &[u16] = &[{sync_terms_src}];

/// ACTION 조회 결과(내부 전달)
#[derive(Copy, Clone, Debug)]
pub struct Action {{ pub sym: u16, pub kind: i8, pub arg: i32 }}  // kind: 1=shift, 2=reduce, 3=accept

/// 프로덕션 메타데이터(축약용)
/// - `PROD_LHS[p]`    : p번 프로덕션의 좌변 **비단말 로컬 인덱스** (0..N_NONTERMS-1)
/// - `PROD_RHS_LEN[p]`: p번 프로덕션의 우변 길이(스택 POP 개수)
pub static PROD_LHS: &[u16] = &[{prod_lhs_src}];
pub static PROD_RHS_LEN: &[u16] = &[{prod_rhs_len_src}];
"""

    actions_src = _fmt_action_dense(ir)
    gotos_src = _fmt_goto_dense(ir)

    terms_src = ", ".join(f"\"{_escape_rs(t)}\"" for t in ir.terms)
    nonterms_src = ", ".join(f"\"{_escape_rs(nt)}\"" for nt in ir.nonterms)
    names_src = f"""\
/// 심볼 이름 디버그 테이블
pub static TERM_NAMES: &[&str] = &[{terms_src}];
pub static NONTERM_NAMES: &[&str] = &[{nonterms_src}];
"""

    # 고속 런타임 + 오류 복구 + 액션 런타임
    runtime = r"""
/// 사용자가 제공해야 하는 **렉서 인터페이스**,
/// 파서는 토큰의 **종류 ID(u16)** 만 필요
pub trait Lexer {
    /// 다음 토큰의 (단말 ID)를 반환. EOF면 Some(EOF_TERM) 또는 None을 반환.
    fn next_kind(&mut self) -> Option<u16>;
}

#[derive(Debug)]
/// 구문 오류 보고용 구조체.
pub struct ParseError {
    pub state: usize,
    pub lookahead: u16, // 단말 ID
}

#[inline]
fn find_action(state: usize, term: u16) -> Option<Action> {
    let idx = state * N_TERMS + (term as usize);
    let k = ACTION_KIND[idx];
    if k == 0 { return None; } // 0=None
    let a = ACTION_ARG[idx];
    Some(Action { sym: term, kind: k, arg: a })
}

#[inline]
fn goto_state(state: usize, nonterm: u16) -> Option<usize> {
    let idx = state * N_NONTERMS + (nonterm as usize);
    let s = GOTO_NEXT[idx];
    if s < 0 { None } else { Some(s as usize) }
}

#[inline]
fn is_sync_term(t: u16) -> bool {
    for &x in SYNC_TERMS.iter() {
        if x == t { return true; }
    }
    false
}

/// LALR/SLR 공용 **LR 파서 루프(인식 전용)**.
pub fn parse<L: Lexer>(mut lx: L) -> Result<(), ParseError> {
    let mut stack: Vec<usize> = vec![START_STATE];
    let mut la: u16 = match lx.next_kind() { Some(k) => k, None => EOF_TERM };
    let mut error_count: usize = 0;

    loop {
        let s = *stack.last().unwrap();
        let act = find_action(s, la);

        match act {
            None => {
                if !(RECOVER_PANIC) || SYNC_TERMS.is_empty() {
                    return Err(ParseError { state: s, lookahead: la });
                }
                // PANIC 모드: 입력/스택 동기화
                error_count += 1;
                if error_count > MAX_ERRORS || la == EOF_TERM {
                    return Err(ParseError { state: s, lookahead: la });
                }
                // 1) 입력을 sync 토큰까지 스킵
                while la != EOF_TERM && !is_sync_term(la) {
                    la = match lx.next_kind() { Some(k) => k, None => EOF_TERM };
                }
                if la == EOF_TERM {
                    return Err(ParseError { state: s, lookahead: la });
                }
                // 2) 스택 팝: (state, la)에 액션이 생길 때까지
                loop {
                    let cur = *stack.last().unwrap();
                    if find_action(cur, la).is_some() { break; }
                    if stack.len() <= 1 { break; } // 시작 상태까지만
                    let _ = stack.pop();
                }
                // 3) 그래도 액션이 없으면 sync 토큰을 하나 소비하고 계속
                let cur = *stack.last().unwrap();
                if find_action(cur, la).is_none() {
                    la = match lx.next_kind() { Some(k) => k, None => EOF_TERM };
                }
                continue;
            }
            Some(Action{kind:1, arg, ..}) => { // shift
                stack.push(arg as usize);
                la = match lx.next_kind() { Some(k) => k, None => EOF_TERM };
            }
            Some(Action{kind:2, arg, ..}) => { // reduce
                let p = arg as usize;
                let popn = PROD_RHS_LEN[p] as usize;
                for _ in 0..popn { let _ = stack.pop(); }
                let t = *stack.last().unwrap();
                let lhs = PROD_LHS[p] as u16;
                let ns = goto_state(t, lhs).expect("GOTO missing");
                stack.push(ns);
            }
            Some(Action{kind:3, ..}) => { // accept
                return Ok(());
            }
            _ => unreachable!()
        }
    }
}


/// =============================
///   액션 기반 값 생성 런타임
/// =============================

/// 토큰 전체를 전달하는 타입(액션 모드 전용)
#[derive(Clone, Debug)]
pub struct Token {
    pub kind: u16,
    pub text: String,
}

/// 전체 토큰을 제공하는 렉서(액션 모드 전용)
pub trait Lexer2 {
    /// 다음 토큰을 반환. EOF면 None
    fn next(&mut self) -> Option<Token>;
}

/// 세만틱 액션 콜백 집합
pub trait Actions {
    type Val;
    /// shift 시: 토큰으로부터 값 생성 (예: NUMBER.lexeme → f64)
    fn on_shift(&mut self, tok: &Token) -> Self::Val;
    /// reduce 시: p(프로덕션 인덱스), lhs(로컬 비단말 id), rhs 값 슬라이스
    fn on_reduce(&mut self, p: usize, lhs: u16, rhs: &[Self::Val]) -> Self::Val;
}

#[inline]
fn pop_one_frame<V>(state_stack: &mut Vec<usize>, val_stack: &mut Vec<V>) {
    // 상태 하나를 팝할 때 값 스택도 1개 팝하여 불변식 len(val)=len(state)-1 유지
    if !state_stack.is_empty() {
        let _ = state_stack.pop();
        if val_stack.len() >= state_stack.len() {
            let _ = val_stack.pop();
        }
    }
}

/// 값 생성 파서: 최종 LHS 값을 반환
pub fn parse_with_actions<L: Lexer2, A: Actions>(mut lx: L, act: &mut A)
    -> Result<A::Val, ParseError>
{
    let mut state_stack: Vec<usize> = vec![START_STATE];
    let mut val_stack: Vec<A::Val> = Vec::new();
    let mut error_count: usize = 0;

    // lookahead 초기화
    let mut la: Token = match lx.next() {
        Some(t) => t,
        None => Token{ kind: EOF_TERM, text: String::new() },
    };

    loop {
        let s = *state_stack.last().unwrap();
        let action = find_action(s, la.kind);

        match action {
            None => {
                if !(RECOVER_PANIC) || SYNC_TERMS.is_empty() {
                    return Err(ParseError { state: s, lookahead: la.kind });
                }
                // PANIC 모드
                error_count += 1;
                if error_count > MAX_ERRORS || la.kind == EOF_TERM {
                    return Err(ParseError { state: s, lookahead: la.kind });
                }
                // 1) 입력을 sync 토큰까지 스킵
                while la.kind != EOF_TERM && !is_sync_term(la.kind) {
                    la = match lx.next() { Some(t) => t, None => Token { kind: EOF_TERM, text: String::new() } };
                }
                if la.kind == EOF_TERM {
                    return Err(ParseError { state: s, lookahead: la.kind });
                }
                // 2) 스택 팝: 액션이 가능할 때까지
                loop {
                    let cur = *state_stack.last().unwrap();
                    if find_action(cur, la.kind).is_some() { break; }
                    if state_stack.len() <= 1 { break; }
                    pop_one_frame(&mut state_stack, &mut val_stack);
                }
                // 3) 여전히 액션이 없으면 sync 토큰을 하나 소비
                let cur = *state_stack.last().unwrap();
                if find_action(cur, la.kind).is_none() {
                    la = match lx.next() { Some(t) => t, None => Token { kind: EOF_TERM, text: String::new() } };
                }
                continue;
            }
            Some(Action{kind:1, arg, ..}) => { // SHIFT
                let v = act.on_shift(&la);
                val_stack.push(v);
                state_stack.push(arg as usize);
                la = match lx.next() { Some(t) => t, None => Token{ kind: EOF_TERM, text: String::new() } };
            }
            Some(Action{kind:2, arg, ..}) => { // REDUCE
                let p = arg as usize;
                let n = PROD_RHS_LEN[p] as usize;
                let lhs = PROD_LHS[p] as u16;

                let len = val_stack.len();
                let rhs_slice = if n == 0 { &[] } else { &val_stack[len - n .. len] };
                let new_val = act.on_reduce(p, lhs, rhs_slice);

                for _ in 0..n {
                    let _ = state_stack.pop();
                    let _ = val_stack.pop();
                }

                let t = *state_stack.last().unwrap();
                let ns = goto_state(t, lhs).expect("GOTO missing");
                state_stack.push(ns);
                val_stack.push(new_val);
            }
            Some(Action{kind:3, ..}) => { // ACCEPT
                if let Some(v) = val_stack.pop() {
                    return Ok(v);
                } else {
                    panic!("accept with empty value stack; provide a value in zero-length reduce");
                }
            }
            _ => unreachable!(),
        }
    }
}
"""

    lexer_src = _lexer_src_from_ast(lexer_ast) if lexer_ast is not None else ""

    return "\n".join([header, prods_src, actions_src, gotos_src, names_src, runtime, lexer_src])
