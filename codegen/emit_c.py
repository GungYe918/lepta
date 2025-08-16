# lepta/codegen/emit_c.py
"""C Code Emit (단일 .c 파일 생성; 런타임 포함).

개요
----
- CodegenIR을 받아 C 소스 코드를 **문자열로** 생성한다.
- 방출되는 C 파일은:
  * 테이블 상수(ACTION/GOTO/PROD 등)
  * 고속 파서 런타임(스택 머신)
  * 최소한의 API (Lexer 인터페이스 + parse 함수)
  * (+옵션) 간단 토크나이저(SimpleLexer)         ← lexer_ast 인자 전달 시 포함
  * 액션 기반 값 생성 런타임(parse_with_actions)
  * (+옵션) 오류 복구(PANIC 모드): %recover / %sync 지시어 사용 시 활성화

성능 개선
---------
- (구) MVP 런타임은 상태별 ACTION/GOTO를 **선형 탐색**으로 찾았다.
- (신) 본 버전은 ACTION/GOTO를 **Dense 2D 테이블(행렬)** 로 방출하여
  **상수시간 O(1)** 조회가 가능하다.

  * ACTION_KIND: int8_t[S * T]  (0=None, 1=shift, 2=reduce, 3=accept)
  * ACTION_ARG : int32_t[S * T] (shift의 next_state 또는 reduce의 prod_idx)
  * GOTO_NEXT  : int32_t[S * N] (-1=None, 그 외 next_state)

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
    """정수 리스트를 C 소스 상수 배열로 직렬화한다."""
    return ", ".join(str(x) for x in ints)


def _preflight_check(ir: CodegenIR) -> None:
    """기본 불변식/전제 조건을 조기 검증하여 생성 단계에서 실패시킨다."""
    if ir.n_states <= 0 or ir.n_terms <= 0 or ir.n_nonterms <= 0:
        raise ValueError(f"emit_c: invalid sizes (states={ir.n_states}, terms={ir.n_terms}, nonterms={ir.n_nonterms})")
    if len(ir.terms) != ir.n_terms:
        raise ValueError(f"emit_c: len(terms)={len(ir.terms)} != n_terms={ir.n_terms}")
    if ir.terms[0] != "$":
        raise ValueError("emit_c: TERM_NAMES[0] must be '$' (EOF)")
    if len(ir.nonterms) != ir.n_nonterms:
        raise ValueError(f"emit_c: len(nonterms)={len(ir.nonterms)} != n_nonterms={ir.n_nonterms}")
    if len(ir.prod_lhs) != len(ir.prod_rhs_len):
        raise ValueError("emit_c: PROD_LHS/PROD_RHS_LEN length mismatch")


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
                f"emit_c: PROD_LHS[{i}] invalid lhs={v} "
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
                    f"emit_c: PROD_LHS uses mixed schemes (found local {v} with global scheme); index={i}"
                )
    else:
        for i, v in enumerate(vals):
            if 0 <= v < NNT:
                out.append(v)
            else:
                raise ValueError(
                    f"emit_c: PROD_LHS uses mixed schemes (found global-like {v} with local scheme); index={i}"
                )

    assert all(0 <= v < NNT for v in out), "emit_c: normalized PROD_LHS out of range"
    return out


def _fmt_action_dense(ir: CodegenIR) -> str:
    """
    ACTION을 Dense 2D 테이블로 직렬화한다.
    - KIND: 0=None, 1=shift, 2=reduce, 3=accept
    - ARG : shift의 다음 상태 또는 reduce의 프로덕션 인덱스
    """
    S = ir.n_states
    T = ir.n_terms
    kind = [0] * (S * T)     # int8_t
    arg  = [0] * (S * T)     # int32_t

    for s, row in enumerate(ir.action_rows):
        base = s * T
        for (term, k, a) in row:
            if not (0 <= term < T):
                raise ValueError(f"emit_c: ACTION term out of range: term={term}, n_terms={T}, state={s}")
            if k not in (SHIFT, REDUCE, ACCEPT):
                raise ValueError(f"emit_c: ACTION kind invalid: kind={k}, expected in {{SHIFT({SHIFT}), REDUCE({REDUCE}), ACCEPT({ACCEPT})}}")
            idx = base + term
            kind[idx] = int(k)
            arg[idx]  = int(a)

    if len(kind) != S * T or len(arg) != S * T:
        raise AssertionError("emit_c: ACTION dense table size mismatch")

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
/// 해당 ID의 문자열 이름은 `TERM_NAMES[term]`로 확인할 수 있다.
static const int8_t ACTION_KIND[] = {{{kind_src}}};
static const int32_t ACTION_ARG[]  = {{{arg_src}}};
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
        raise ValueError(f"emit_c: GOTO nonterm id out of range: {bad[:5]}...")

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
                        f"emit_c: GOTO mixed schemes (found local aid={aid} with global scheme); state={s}"
                    )
            else:
                if 0 <= aid < N:
                    a_local = aid
                else:
                    raise ValueError(
                        f"emit_c: GOTO mixed schemes (found global-like aid={aid} with local scheme); state={s}"
                    )

            if not (0 <= ns < S):
                raise ValueError(
                    f"emit_c: GOTO next state out of range: ns={ns}, n_states={S}, state={s}, A_local={a_local}"
                )

            idx = base + a_local
            nexts[idx] = int(ns)

    if len(nexts) != S * N:
        raise AssertionError("emit_c: GOTO dense table size mismatch")

    nxt_src = ", ".join(str(x) for x in nexts)
    return f"""\
/// Dense GOTO 테이블 (O(1) 조회).
/// 인덱스 계산: `idx = state * N_NONTERMS + nonterm`
/// - `GOTO_NEXT[idx]` < 0 : 이동 없음(잘못된 축약/테이블 생성 오류)
/// - `GOTO_NEXT[idx]` >= 0: 다음 상태 번호
///
/// reduce 후, 스택 top 상태를 t라고 할 때:
/// `goto_state(t, PROD_LHS[p])`가 새로운 상태를 제공한다.
static const int32_t GOTO_NEXT[] = {{{nxt_src}}};
"""


def _escape_c(s: str) -> str:
    """C 문자열 리터럴 이스케이프."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _lexer_src_from_ast(ast) -> str:
    """
    매우 단순한 **내장 C 렉서**(옵션) 방출:
    - 키워드(리터럴) **길이 내림차순(최장일치)** + 유니코드 단어 경계(단순화)
    - %ignore: `/\\s+/` 가 하나라도 있으면 공백/개행 스킵
    - %token: NUMBER(테스트 문법의 숫자 패턴)만 핸드코딩 스캔 지원
      그 외 %token이 있으면 런타임 에러 발생 (문서화)
    """
    # --- 키워드: 중복 제거 + 길이 내림차순(동길이면 선언 순서) ---
    raw_kws = [kw.lexeme for kw in ast.decl_keywords]
    seen = set()
    kws_pairs = []
    for idx, lit in enumerate(raw_kws):
        if lit in seen:
            continue
        seen.add(lit)
        kws_pairs.append((lit, idx))
    kws_pairs.sort(key=lambda p: (-len(p[0]), p[1]))
    kws = [lit for (lit, _idx) in kws_pairs]
    kws_arr = ", ".join(f"\"{_escape_c(k)}\"" for k in kws)

    # ignore: /\s+/ 존재 여부
    has_ws_ignore = any(ig.pattern == r"\s+" for ig in ast.decl_ignores)

    # token 이름 수집
    token_names = [td.name for td in ast.decl_tokens]
    has_number = "NUMBER" in token_names

    # skip_ignores
    if has_ws_ignore:
        skip_ignores_code = r"""
        while (self->i < self->len) {
            unsigned char c = (unsigned char)self->text[self->i];
            if (c==' '||c=='\n'||c=='\r'||c=='\t'||c=='\v'||c=='\f') {
                self->i++;
            } else {
                break;
            }
        }
        """.strip("\n")
    else:
        skip_ignores_code = "/* no %ignore /\\s+/; */"

    # match_number
    if has_number:
        match_number_code = r"""
        size_t start = self->i;

        // optional sign
        if (self->i < self->len && self->text[self->i] == '-') self->i++;

        // int part
        if (self->i < self->len && self->text[self->i] == '0') {
            self->i++;
        } else {
            if (self->i < self->len) {
                unsigned char ch = (unsigned char)self->text[self->i];
                if (ch >= '1' && ch <= '9') {
                    self->i++;
                    while (self->i < self->len) {
                        unsigned char c = (unsigned char)self->text[self->i];
                        if ((c >= '0' && c <= '9') || c == '_') self->i++; else break;
                    }
                } else {
                    self->i = start;
                    return 0;
                }
            } else {
                self->i = start;
                return 0;
            }
        }

        // frac
        if (self->i < self->len && self->text[self->i] == '.') {
            self->i++;
            size_t before = self->i;
            while (self->i < self->len) {
                unsigned char c = (unsigned char)self->text[self->i];
                if ((c >= '0' && c <= '9') || c == '_') self->i++; else break;
            }
            if (self->i == before) { self->i = start; return 0; }
        }

        // exp
        if (self->i < self->len) {
            unsigned char ch = (unsigned char)self->text[self->i];
            if (ch == 'e' || ch == 'E') {
                self->i++;
                if (self->i < self->len) {
                    unsigned char sign = (unsigned char)self->text[self->i];
                    if (sign == '+' || sign == '-') self->i++;
                }
                size_t before = self->i;
                while (self->i < self->len) {
                    unsigned char c = (unsigned char)self->text[self->i];
                    if ((c >= '0' && c <= '9') || c == '_') self->i++; else break;
                }
                if (self->i == before) { self->i = start; return 0; }
            }
        }
        return 1;
        """.strip("\n")
    else:
        match_number_code = "return 0;"

    return f"""
// ====== Generated Simple Lexer (no external libs) ======
// 제한: 공백 스킵(\\s+), 키워드 리터럴(최장일치), NUMBER 토큰만 지원합니다.
// 다른 %token 정규식은 지원하지 않습니다. 필요시 직접 Lexer를 구현하세요.

typedef struct {{
    const char* text;
    size_t len;
    size_t i;
}} SimpleLexer;

static SimpleLexer SimpleLexer_new(const char* input) {{
    SimpleLexer lx; lx.text = input; lx.len = (size_t)strlen(input); lx.i = 0; return lx;
}}

static int SL_at_eof(SimpleLexer* self) {{ return self->i >= self->len; }}

static int SL_starts_with_at(SimpleLexer* self, const char* s) {{
    size_t n = strlen(s);
    if (self->i + n > self->len) return 0;
    return memcmp(self->text + self->i, s, n) == 0;
}}

static int SL_is_word_kw(const char* s) {{
    // 매우 단순화된 '단어 키워드' 판정: [A-Za-z0-9_]
    for (const char* p = s; *p; ++p) {{
        unsigned char c = (unsigned char)*p;
        if ((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_') return 1;
    }}
    return 0;
}}

static const char* SL_match_keyword(SimpleLexer* self) {{
    // KEYWORDS는 **길이 내림차순(최장일치)** 로 정렬되어 방출됩니다.
    static const char* KEYWORDS[] = {{{kws_arr}}};
    static const size_t KEYWORDS_LEN = sizeof(KEYWORDS)/sizeof(KEYWORDS[0]);

    for (size_t i = 0; i < KEYWORDS_LEN; ++i) {{
        const char* kw = KEYWORDS[i];
        size_t kwlen = strlen(kw);
        if (SL_starts_with_at(self, kw)) {{
            if (SL_is_word_kw(kw)) {{
                // prev boundary
                int prev_ok = 1;
                if (self->i > 0) {{
                    unsigned char prev = (unsigned char)self->text[self->i - 1];
                    if ((prev>='A'&&prev<='Z')||(prev>='a'&&prev<='z')||(prev>='0'&&prev<='9')||prev=='_') prev_ok = 0;
                }}
                // next boundary
                int next_ok = 1;
                size_t after = self->i + kwlen;
                if (after < self->len) {{
                    unsigned char next = (unsigned char)self->text[after];
                    if ((next>='A'&&next<='Z')||(next>='a'&&next<='z')||(next>='0'&&next<='9')||next=='_') next_ok = 0;
                }}
                if (!(prev_ok && next_ok)) continue;
            }}
            self->i += kwlen;
            return kw;
        }}
    }}
    return NULL;
}}

static void SL_skip_ignores(SimpleLexer* self) {{
    {skip_ignores_code}
}}

static int SL_match_number(SimpleLexer* self) {{
    {match_number_code}
}}

// Lexer 인터페이스 어댑터: TERM_NAMES를 순회해 kind를 찾는다.
static unsigned short SimpleLexer_next_kind(SimpleLexer* self) {{
    SL_skip_ignores(self);
    if (SL_at_eof(self)) return 0; // EOF_TERM

    const char* kw = SL_match_keyword(self);
    if (kw) {{
        for (size_t i = 0; i < N_TERMS; ++i) {{
            if (strcmp(TERM_NAMES[i], kw) == 0) return (unsigned short)i;
        }}
        fprintf(stderr, "keyword not found in TERM_NAMES: %s\\n", kw);
        abort();
    }}

    if (SL_match_number(self)) {{
        for (size_t i = 0; i < N_TERMS; ++i) {{
            if (strcmp(TERM_NAMES[i], "NUMBER") == 0) return (unsigned short)i;
        }}
        fprintf(stderr, "token not found in TERM_NAMES: NUMBER\\n");
        abort();
    }}

    unsigned char bad = (unsigned char)self->text[self->i];
    fprintf(stderr, "Lexing error: unsupported token starting with '%c' at byte %zu\\n", bad, self->i);
    abort();
}}
""".lstrip("\n")


# ---------- 메인 방출기 ----------

def emit_c_to_string(ir: CodegenIR, module_name: str = "parser", *, lexer_ast: Optional[object] = None) -> str:
    """
    emit_c_to_string(ir, module_name, lexer_ast=None) -> str
    --------------------------------------------------------
    CodegenIR을 받아 **하나의 C 소스 문자열**을 생성한다.
    lexer_ast 가 주어지면, 간단 C 렉서(SimpleLexer)도 함께 방출한다.
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
//! - 사용 방법: `#include "{module_name}.h"` 또는 동일한 심볼 가시성
//! - 필요 요소: `Lexer` 인터페이스 구현(다음 토큰의 **종류 ID**를 돌려줌)
//! - 파싱 결과: 성공 시 0, 실패 시 음수 및 ParseError 채움
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
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
"""

    # 메타 + 프로덕션
    prod_lhs_src = _fmt_list_int(lhs_local)
    prod_rhs_len_src = _fmt_list_int(ir.prod_rhs_len)
    prods_src = f"""\
/// 파서 테이블 메타데이터
static const size_t N_STATES = {ir.n_states};
static const size_t N_TERMS  = {ir.n_terms};
static const size_t N_NONTERMS = {ir.n_nonterms};
static const size_t START_STATE = {ir.start_state};
static const uint16_t EOF_TERM = 0; // terms[0] == "$"

/// 오류 복구 메타
static const int RECOVER_PANIC = {1 if recover_panic else 0};
static const size_t MAX_ERRORS = {max_err};
static const uint16_t SYNC_TERMS[] = {{{sync_terms_src}}};
static const size_t SYNC_TERMS_LEN = sizeof(SYNC_TERMS)/sizeof(SYNC_TERMS[0]);

/// ACTION 조회 결과(내부 전달)
/// kind: 1=shift, 2=reduce, 3=accept
typedef struct {{ uint16_t sym; int8_t kind; int32_t arg; }} Action;

/// 프로덕션 메타데이터(축약용)
/// - `PROD_LHS[p]`    : p번 프로덕션의 좌변 **비단말 로컬 인덱스** (0..N_NONTERMS-1)
/// - `PROD_RHS_LEN[p]`: p번 프로덕션의 우변 길이(스택 POP 개수)
static const uint16_t PROD_LHS[] = {{{prod_lhs_src}}};
static const uint16_t PROD_RHS_LEN[] = {{{prod_rhs_len_src}}};
"""

    actions_src = _fmt_action_dense(ir)
    gotos_src = _fmt_goto_dense(ir)

    terms_src = ", ".join(f"\"{_escape_c(t)}\"" for t in ir.terms)
    nonterms_src = ", ".join(f"\"{_escape_c(nt)}\"" for nt in ir.nonterms)
    names_src = f"""\
/// 심볼 이름 디버그 테이블
static const char* const TERM_NAMES[] = {{{terms_src}}};
static const char* const NONTERM_NAMES[] = {{{nonterms_src}}};
"""

    # 고속 런타임 + 오류 복구 + 액션 런타임
    runtime = r"""
/// 사용자가 제공해야 하는 **렉서 인터페이스**,
/// 파서는 토큰의 **종류 ID(u16)** 만 필요
typedef struct {
    // 다음 토큰의 (단말 ID)를 반환. EOF면 EOF_TERM을 반환.
    uint16_t (*next_kind)(void* ctx);
    void* ctx;
} Lexer;


/// 구문 오류 보고용 구조체.
typedef struct {
    size_t state;
    uint16_t lookahead; // 단말 ID
} ParseError;

static inline int has_action(size_t state, uint16_t term) {
    size_t idx = state * N_TERMS + (size_t)term;
    int8_t k = ACTION_KIND[idx];
    return k != 0;
}

static inline int find_action(size_t state, uint16_t term, Action* out) {
    size_t idx = state * N_TERMS + (size_t)term;
    int8_t k = ACTION_KIND[idx];
    if (k == 0) return 0; // 0=None
    int32_t a = ACTION_ARG[idx];
    out->sym = term; out->kind = k; out->arg = a;
    return 1;
}

static inline int goto_state(size_t state, uint16_t nonterm, size_t* out) {
    size_t idx = state * N_NONTERMS + (size_t)nonterm;
    int32_t s = GOTO_NEXT[idx];
    if (s < 0) return 0;
    *out = (size_t)s;
    return 1;
}

static inline int is_sync_term(uint16_t t) {
    for (size_t i = 0; i < SYNC_TERMS_LEN; ++i) {
        if (SYNC_TERMS[i] == t) return 1;
    }
    return 0;
}

/// LALR/SLR 공용 **LR 파서 루프(인식 전용)**.
int parse(Lexer lx, ParseError* err_out) {
    size_t cap = 64, len = 0;
    size_t* stack = (size_t*)malloc(sizeof(size_t)*cap);
    if (!stack) { fprintf(stderr, "oom\\n"); return -1; }
    stack[len++] = START_STATE;

    uint16_t la = lx.next_kind(lx.ctx);

    size_t error_count = 0;

    for (;;) {
        size_t s = stack[len-1];
        Action act;
        int ok = find_action(s, la, &act);

        if (!ok) {
            if (!(RECOVER_PANIC) || SYNC_TERMS_LEN == 0) {
                if (err_out) { err_out->state = s; err_out->lookahead = la; }
                free(stack);
                return -1;
            }
            // PANIC 모드: 입력/스택 동기화
            error_count += 1;
            if (error_count > MAX_ERRORS || la == EOF_TERM) {
                if (err_out) { err_out->state = s; err_out->lookahead = la; }
                free(stack);
                return -1;
            }
            // 1) 입력을 sync 토큰까지 스킵
            while (la != EOF_TERM && !is_sync_term(la)) {
                la = lx.next_kind(lx.ctx);
            }
            if (la == EOF_TERM) {
                if (err_out) { err_out->state = s; err_out->lookahead = la; }
                free(stack);
                return -1;
            }
            // 2) 스택 팝: (state, la)에 액션이 생길 때까지
            for (;;) {
                size_t cur = stack[len-1];
                if (has_action(cur, la)) break;
                if (len <= 1) break;
                --len; // pop
            }
            // 3) 그래도 액션이 없으면 sync 토큰 하나 소비
            {
                size_t cur = stack[len-1];
                Action tmp;
                if (!find_action(cur, la, &tmp)) {
                    la = lx.next_kind(lx.ctx);
                }
            }
            continue;
        }

        if (act.kind == 1) { // shift
            if (len >= cap) { cap *= 2; stack = (size_t*)realloc(stack, sizeof(size_t)*cap); if (!stack) { fprintf(stderr,"oom\\n"); return -1; } }
            stack[len++] = (size_t)act.arg;
            la = lx.next_kind(lx.ctx);
        } else if (act.kind == 2) { // reduce
            size_t p = (size_t)act.arg;
            size_t popn = (size_t)PROD_RHS_LEN[p];
            for (size_t i = 0; i < popn; ++i) { if (len>0) --len; }
            size_t t = stack[len-1];
            uint16_t lhs = (uint16_t)PROD_LHS[p];
            size_t ns;
            if (!goto_state(t, lhs, &ns)) { fprintf(stderr, "GOTO missing\\n"); free(stack); return -1; }
            if (len >= cap) { cap *= 2; stack = (size_t*)realloc(stack, sizeof(size_t)*cap); if (!stack) { fprintf(stderr,"oom\\n"); return -1; } }
            stack[len++] = ns;
        } else if (act.kind == 3) { // accept
            free(stack);
            return 0;
        } else {
            abort();
        }
    }
}


/// =============================
///   액션 기반 값 생성 런타임
/// =============================

/// 토큰 전체를 전달하는 타입(액션 모드 전용)
typedef struct {
    uint16_t kind;
    char* text; // 소유권 규약은 사용자 구현에 따름
} Token;

/// 전체 토큰을 제공하는 렉서(액션 모드 전용)
typedef struct {
    Token* (*next)(void* ctx); // EOF면 NULL
    void* ctx;
} Lexer2;

/// 세만틱 액션 콜백 집합
typedef struct {
    // user는 사용자 상태 포인터
    void* user;
    // shift 시: 토큰으로부터 값 생성 (예: NUMBER.lexeme → malloc된 값 포인터)
    void* (*on_shift)(void* user, const Token* tok);
    // reduce 시: p(프로덕션 인덱스), lhs(로컬 비단말 id), rhs 값 배열/길이
    void* (*on_reduce)(void* user, size_t p, uint16_t lhs, void** rhs, size_t rhs_len);
} Actions;

static inline void pop_one_frame(size_t* len_states, size_t* len_vals, size_t min_keep) {
    if (*len_states > min_keep) {
        --(*len_states);
        if (*len_vals >= *len_states) --(*len_vals);
    }
}

/// 값 생성 파서: 최종 LHS 값을 반환(성공 시 non-NULL, 실패 시 NULL; 오류는 err_out)
void* parse_with_actions(Lexer2 lx, Actions* act, ParseError* err_out) {
    // 상태 스택
    size_t cap_s = 64, len_s = 0;
    size_t* state_stack = (size_t*)malloc(sizeof(size_t)*cap_s);
    if (!state_stack) { fprintf(stderr, "oom\\n"); return NULL; }
    state_stack[len_s++] = START_STATE;

    // 값 스택
    size_t cap_v = 64, len_v = 0;
    void** val_stack = (void**)malloc(sizeof(void*)*cap_v);
    if (!val_stack) { free(state_stack); fprintf(stderr, "oom\\n"); return NULL; }

    size_t error_count = 0;

    // lookahead 초기화
    Token* la = lx.next(lx.ctx);
    Token eof_tok = { EOF_TERM, NULL };
    if (!la) la = &eof_tok;

    for (;;) {
        size_t s = state_stack[len_s-1];
        Action action;
        int ok = find_action(s, la->kind, &action);

        if (!ok) {
            if (!(RECOVER_PANIC) || SYNC_TERMS_LEN == 0) {
                if (err_out) { err_out->state = s; err_out->lookahead = la->kind; }
                free(val_stack); free(state_stack);
                return NULL;
            }
            // PANIC 모드
            error_count += 1;
            if (error_count > MAX_ERRORS || la->kind == EOF_TERM) {
                if (err_out) { err_out->state = s; err_out->lookahead = la->kind; }
                free(val_stack); free(state_stack);
                return NULL;
            }
            // 1) 입력을 sync 토큰까지 스킵
            while (la->kind != EOF_TERM && !is_sync_term(la->kind)) {
                la = lx.next(lx.ctx);
                if (!la) la = &eof_tok;
            }
            if (la->kind == EOF_TERM) {
                if (err_out) { err_out->state = s; err_out->lookahead = la->kind; }
                free(val_stack); free(state_stack);
                return NULL;
            }
            // 2) 스택 팝: 액션이 가능할 때까지
            for (;;) {
                size_t cur = state_stack[len_s-1];
                if (has_action(cur, la->kind)) break;
                if (len_s <= 1) break;
                pop_one_frame(&len_s, &len_v, 1);
            }
            // 3) 여전히 액션이 없으면 sync 토큰을 하나 소비
            {
                size_t cur = state_stack[len_s-1];
                Action tmp;
                if (!find_action(cur, la->kind, &tmp)) {
                    la = lx.next(lx.ctx);
                    if (!la) la = &eof_tok;
                }
            }
            continue;
        }

        if (action.kind == 1) { // SHIFT
            if (len_v >= cap_v) { cap_v *= 2; val_stack = (void**)realloc(val_stack, sizeof(void*)*cap_v); if (!val_stack) { fprintf(stderr, "oom\\n"); free(state_stack); return NULL; } }
            void* v = act->on_shift(act->user, la);
            val_stack[len_v++] = v;

            if (len_s >= cap_s) { cap_s *= 2; state_stack = (size_t*)realloc(state_stack, sizeof(size_t)*cap_s); if (!state_stack) { fprintf(stderr, "oom\\n"); free(val_stack); return NULL; } }
            state_stack[len_s++] = (size_t)action.arg;

            la = lx.next(lx.ctx);
            if (!la) la = &eof_tok;
        } else if (action.kind == 2) { // REDUCE
            size_t p = (size_t)action.arg;
            size_t n = (size_t)PROD_RHS_LEN[p];
            uint16_t lhs = (uint16_t)PROD_LHS[p];

            void** rhs_slice = NULL;
            if (n > 0) rhs_slice = &val_stack[len_v - n];
            void* new_val = act->on_reduce(act->user, p, lhs, rhs_slice, n);

            for (size_t i = 0; i < n; ++i) { if (len_s>0) --len_s; if (len_v>0) --len_v; }

            size_t t = state_stack[len_s-1];
            size_t ns;
            if (!goto_state(t, lhs, &ns)) { fprintf(stderr, "GOTO missing\\n"); free(val_stack); free(state_stack); return NULL; }
            if (len_s >= cap_s) { cap_s *= 2; state_stack = (size_t*)realloc(state_stack, sizeof(size_t)*cap_s); if (!state_stack) { fprintf(stderr, "oom\\n"); free(val_stack); return NULL; } }
            state_stack[len_s++] = ns;

            if (len_v >= cap_v) { cap_v *= 2; val_stack = (void**)realloc(val_stack, sizeof(void*)*cap_v); if (!val_stack) { fprintf(stderr, "oom\\n"); free(state_stack); return NULL; } }
            val_stack[len_v++] = new_val;
        } else if (action.kind == 3) { // ACCEPT
            void* out = NULL;
            if (len_v > 0) out = val_stack[--len_v];
            else { fprintf(stderr, "accept with empty value stack; provide a value in zero-length reduce\\n"); }
            free(val_stack); free(state_stack);
            return out;
        } else {
            abort();
        }
    }
}
"""

    lexer_src = _lexer_src_from_ast(lexer_ast) if lexer_ast is not None else ""

    return "\n".join([header, prods_src, actions_src, gotos_src, names_src, runtime, lexer_src])
