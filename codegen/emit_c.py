# lepta/codegen/emit_c.py
"""C Code Emit (단일 .c 파일 생성; 런타임 포함).

개요
----
- CodegenIR을 받아 C 소스 코드를 **문자열로** 생성한다.
- 방출되는 C 파일은:
  * 테이블 상수(ACTION/GOTO/PROD 등)
  * 고속 파서 런타임(스택 머신)
  * 최소한의 API (Lexer 콜백 + parse 함수)
  로 구성된다.

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
from typing import List
from .ir import CodegenIR, SHIFT, REDUCE, ACCEPT


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

    # 스키마 결정: 엄격 글로벌이 하나라도 있으면 글로벌 스키마
    has_strict_global = any(NT < v < NT + NNT for v in vals)
    use_global = has_strict_global

    out: List[int] = []
    if use_global:
        # 글로벌 스키마: v >= NT 는 모두 글로벌로 간주(경계값 포함)
        for i, v in enumerate(vals):
            if v >= NT:
                out.append(v - NT)
            else:
                # 혼재 금지(명확성을 위해 에러)
                raise ValueError(
                    f"emit_c: PROD_LHS uses mixed schemes (found local {v} with global scheme); index={i}"
                )
    else:
        # 로컬 스키마: 모두 로컬 범위여야 함
        for i, v in enumerate(vals):
            if 0 <= v < NNT:
                out.append(v)
            else:
                raise ValueError(
                    f"emit_c: PROD_LHS uses mixed schemes (found global-like {v} with local scheme); index={i}"
                )

    # 최종 범위 보증
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
/// 참고: term은 `Lexer::next_kind()`에 해당하는 **단말 ID(uint16_t)** 이어야 하며,
/// 해당 ID의 문자열 이름은 `TERM_NAMES[term]`로 확인할 수 있다.
static const int8_t ACTION_KIND[] = {{ {kind_src} }};
static const int32_t ACTION_ARG[]  = {{ {arg_src} }};
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

    # 스키마 판정: 전체 aid를 스캔해서 결정
    all_aids = [aid for row in ir.goto_rows for (aid, _ns) in row]
    if not all(0 <= a < N or NT <= a < NT + N for a in all_aids):
        bad = [a for a in all_aids if not (0 <= a < N or NT <= a < NT + N)]
        raise ValueError(f"emit_c: GOTO nonterm id out of range: {bad[:5]}...")

    has_strict_global = any(NT < a < NT + N for a in all_aids)
    use_global = has_strict_global

    for s, row in enumerate(ir.goto_rows):
        base = s * N
        for (aid, ns) in row:
            # --- aid 정규화 ---
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
static const int32_t GOTO_NEXT[] = {{ {nxt_src} }};
"""


def emit_c_to_string(ir: CodegenIR, module_name: str = "parser") -> str:
    """
    emit_c_to_string(ir, module_name) -> str
    ----------------------------------------
    CodegenIR을 받아 **하나의 C 소스 문자열**을 생성한다.

    Parameters
    ----------
    ir : CodegenIR
        파서 테이블/심볼/프로덕션 정보.
    module_name : str
        최상단 모듈/팩키지 이름(식별자). (정보용 주석)

    Returns
    -------
    str
        C 소스 코드 전체 문자열.
    """
    _preflight_check(ir)
    lhs_local = _normalize_prod_lhs_to_local(ir)  # ★ 핵심: 로컬 비단말 인덱스로 정규화

    header = f"""\
/* Auto-generated by lepta
 * module: {module_name}
 * states={ir.n_states}, terms={ir.n_terms}, nonterms={ir.n_nonterms}
 *
 * 이 파일은 Rasca가 `.g` 문법에서 생성한 **LALR/SLR 파서**의
 * **런타임(스택 머신) + 테이블**을 한 파일에 담은 것이다.
 * - 사용 방법: 이 파일을 프로젝트에 포함하고, `parse` 함수를 호출
 * - 필요 요소: Lexer 콜백(다음 토큰의 **종류 ID**를 돌려줌)
 * - 파싱 결과: 성공 시 0, 실패 시 0 이외 값 반환(+ ParseError에 정보 세팅)
 *
 * ## 테이블 레이아웃(고성능 O(1) 조회)
 * 본 런타임은 **Dense 2D 테이블**을 사용하여 ACTION/GOTO를 O(1)에 조회
 * - ACTION_KIND[s*T + t]: 상태 s에서 단말 t일 때의 동작(0=None, 1=shift, 2=reduce, 3=accept)
 * - ACTION_ARG [s*T + t]: 동작 인자(shift=다음 상태, reduce=프로덕션 인덱스)
 * - GOTO_NEXT  [s*N + A]: 상태 s에서 비단말 A로의 다음 상태(없으면 -1)
 *
 * 메모리 사용량은 S×T, S×N 크기에 비례한다.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {{
#endif

/* 파서 테이블의 기본 메타데이터.
 * - N_STATES   : LR 상태 개수
 * - N_TERMS    : 단말(토큰) 개수 (TERM_NAMES 길이와 동일)
 * - N_NONTERMS : 비단말(구조 심볼) 개수 (NONTERM_NAMES 길이와 동일)
 * - START_STATE: 스택 초기 상태(일반적으로 0)
 * - EOF_TERM   : **반드시 0**. TERM_NAMES[0] == "$" 이며 입력 종료를 의미
 */
enum {{
    N_STATES   = {ir.n_states},
    N_TERMS    = {ir.n_terms},
    N_NONTERMS = {ir.n_nonterms},
    START_STATE = {ir.start_state},
}};
static const uint16_t EOF_TERM = 0; /* terms[0] == "$" */

/* ACTION 조회 결과를 표현하는 경량 구조체.
 * - kind : 1=shift, 2=reduce, 3=accept
 * - arg  : shift일 때 다음 상태, reduce일 때 프로덕션 인덱스
 * 런타임 내부 전달용으로만 사용된다.
 */
typedef struct {{
    uint16_t sym;
    int8_t   kind; /* 1=shift, 2=reduce, 3=accept */
    int32_t  arg;  /* next_state or prod_idx     */
}} Action;

/* 반환 코드: 0=성공, 그 외=실패 */
enum {{ PARSE_OK = 0, PARSE_ERR = 1 }};

"""

    # 프로덕션 테이블 (★ LHS는 로컬 비단말 인덱스로 방출)
    prod_lhs_src = _fmt_list_int(lhs_local)
    prod_rhs_len_src = _fmt_list_int(ir.prod_rhs_len)
    prods_src = f"""\
/// 축약(reduce)에 필요한 프로덕션(규칙) 메타데이터.
/// - `PROD_LHS[p]`    : p번 프로덕션의 좌변 **비단말 로컬 인덱스** (0..N_NONTERMS-1)
/// - `PROD_RHS_LEN[p]`: p번 프로덕션의 우변 길이(스택에서 몇 개를 POP할지 결정)
/// 예) reduce p 수행 시:
///   1) 스택에서 `PROD_RHS_LEN[p]` 만큼 POP
///   2) 남은 top 상태를 t라 할 때, `goto_state(t, PROD_LHS[p])`로 천이
static const uint16_t PROD_LHS[]     = {{ {prod_lhs_src} }};
static const uint16_t PROD_RHS_LEN[] = {{ {prod_rhs_len_src} }};
"""

    # Dense ACTION/GOTO
    actions_src = _fmt_action_dense(ir)
    gotos_src = _fmt_goto_dense(ir)

    # 심볼 이름 디버그 테이블
    def c_str(s: str) -> str:
        """C 문자열 리터럴 이스케이프."""
        return s.replace("\\", "\\\\").replace('"', '\\"')
    terms_src = ", ".join(f"\"{c_str(t)}\"" for t in ir.terms)
    nonterms_src = ", ".join(f"\"{c_str(nt)}\"" for nt in ir.nonterms)
    names_src = f"""\
/// 심볼 이름 디버그 테이블.
/// - `TERM_NAMES[term_id]`   : 단말 이름(예: "$", "(", ")", "NUMBER"...)
/// - `NONTERM_NAMES[nt_id]`  : 비단말 이름(예: "Expr", "Term"...)
/// 디버깅/로그/에러 메시지에서 **사람이 읽을 수 있는 이름**을 복원하는 데 사용
static const char* const TERM_NAMES[]    = {{ {terms_src} }};
static const char* const NONTERM_NAMES[] = {{ {nonterms_src} }};
"""

    # 고속 런타임 (Dense 테이블 O(1) 조회)
    runtime = r"""
/* 사용자가 제공해야 하는 **렉서 인터페이스**,
 * 파서는 토큰의 **종류 ID(uint16_t)** 만 필요
 * 구현 규칙:
 * - 다음 토큰의 종류 ID를 반환한다.
 * - 입력이 끝나면 **반드시 EOF_TERM(=0)** 을 반환하는 것을 권장.
 * - 단말 ID는 `TERM_NAMES` 순서와 일치해야 함.(코드 생성 시 확정).
 *
 * 팁: 개발 초기에 `printf` 등으로 TERM_NAMES를 출력해 ID↔이름을 확인할 것.
 */

/* 간단한 콜백 인터페이스: 사용자가 next_kind를 제공 */
typedef uint16_t (*next_kind_fn)(void* user);
typedef struct Lexer {
    next_kind_fn next_kind;
    void* user; /* 사용자 컨텍스트 포인터 */
} Lexer;

/* 구문 오류 보고용 구조체.
 * - state    : 오류 시점의 LR 상태 번호(에러 복구/분석용)
 * - lookahead: 문제를 일으킨 단말 ID (문자열은 TERM_NAMES[lookahead])
 *
 * 예상 집합을 만들고 싶다면, 상태 s에서 ACTION_KIND[s*T + t] != 0 인 t들의
 * 집합을 모아 메시지로 보여줄 수 있음.
 */
typedef struct ParseError {
    size_t   state;
    uint16_t lookahead; /* 단말 ID */
} ParseError;

/* (상수시간) ACTION 조회.
 * - 입력: LR state, 단말 ID term
 * - 결과: 해당 (state, term)에 대한 동작(Action)을 out에 채움
 * NOTE:
 * - Dense 2D 테이블 인덱스로 직접 접근한다.
 * - 결과가 없으면 0을 반환(파서 루프에서 에러로 처리).
 */
static inline int find_action(size_t state, uint16_t term, Action* out) {
    size_t idx = state * (size_t)N_TERMS + (size_t)term;
    int8_t k = ACTION_KIND[idx];
    if (k == 0) return 0; /* 0=None */
    int32_t a = ACTION_ARG[idx];
    out->sym  = term;
    out->kind = k;
    out->arg  = a;
    return 1;
}

/* (상수시간) GOTO 조회.
 * - 입력: LR state, 비단말 ID nonterm
 * - 결과: 다음 상태(없으면 음수)
 * NOTE:
 * - reduce 후 호출되어 새 상태를 결정함.
 * - 테이블 값이 음수(-1)이면 이동 없음.
 */
static inline int goto_state(size_t state, uint16_t nonterm) {
    size_t idx = state * (size_t)N_NONTERMS + (size_t)nonterm;
    int32_t s = GOTO_NEXT[idx];
    return (int)s; /* 음수면 이동 없음 */
}

/* 내부 스택 보조: capacity 보장 */
static inline int ensure_cap(size_t need, size_t* cap, size_t** stk) {
    if (need <= *cap) return 1;
    size_t new_cap = (*cap == 0 ? 64 : *cap);
    while (new_cap < need) new_cap *= 2;
    void* p = realloc(*stk, new_cap * sizeof(size_t));
    if (!p) return 0;
    *stk = (size_t*)p;
    *cap = new_cap;
    return 1;
}

/* LALR/SLR 공용 **LR 파서 루프**.
 * shift/reduce 스택 머신
 * 동작 방식:
 * 1) 스택에는 **상태 번호**만 적재(초기값 START_STATE).
 * 2) lookahead(다음 토큰 종류)를 구함(EOF면 EOF_TERM).
 * 3) find_action(state, lookahead)로 동작을 결정.
 *    - shift: 스택에 **다음 상태**를 push, lookahead를 다음 토큰으로 갱신
 *    - reduce: p번 프로덕션으로 축약
 *        a) PROD_RHS_LEN[p] 만큼 pop
 *        b) 남은 top 상태를 t라 할 때, goto_state(t, PROD_LHS[p])로 이동하여 push
 *    - accept: 파싱 성공 -> 0 반환
 *    - 없음: 구문 오류 -> ParseError 설정 후 1 반환
 *
 * 주의:
 * - 이 런타임은 **값 계산/세만틱 액션을 수행하지 않습니다**(v1). 필요시 상위 코드에서
 *   토큰 스트림/AST를 별도로 관리하거나, 향후 액션 블록 기능을 사용하세요.
 * - 디버깅 시 TERM_NAMES, NONTERM_NAMES를 함께 출력하면 원인 파악이 쉬워집니다.
 */
static inline uint16_t _next_or_eof(Lexer* lx) {
    uint16_t k = lx->next_kind ? lx->next_kind(lx->user) : EOF_TERM;
    return k;
}

int parse(Lexer* lx, ParseError* err) {
    size_t* stack = NULL;
    size_t  cap = 0, top = 0;

    if (!ensure_cap(1, &cap, &stack)) {
        if (err) { err->state = 0; err->lookahead = EOF_TERM; }
        return PARSE_ERR;
    }
    stack[top++] = (size_t)START_STATE;

    uint16_t la = _next_or_eof(lx);
    for (;;) {
        size_t s = stack[top - 1];
        Action act;
        if (!find_action(s, la, &act)) {
            if (err) { err->state = s; err->lookahead = la; }
            free(stack);
            return PARSE_ERR;
        }

        if (act.kind == 1) { /* shift */
            if (!ensure_cap(top + 1, &cap, &stack)) {
                if (err) { err->state = s; err->lookahead = la; }
                free(stack);
                return PARSE_ERR;
            }
            stack[top++] = (size_t)act.arg;
            la = _next_or_eof(lx);
            continue;
        } else if (act.kind == 2) { /* reduce */
            size_t p = (size_t)act.arg;
            size_t popn = (size_t)PROD_RHS_LEN[p];
            if (popn > top) popn = top; /* 방어적 */
            top -= popn;
            size_t t = stack[top - 1];
            uint16_t lhs = (uint16_t)PROD_LHS[p]; /* ★ 로컬 비단말 인덱스 */
            int ns = goto_state(t, lhs);
            if (ns < 0) {
                if (err) { err->state = t; err->lookahead = la; }
                free(stack);
                return PARSE_ERR;
            }
            if (!ensure_cap(top + 1, &cap, &stack)) {
                if (err) { err->state = t; err->lookahead = la; }
                free(stack);
                return PARSE_ERR;
            }
            stack[top++] = (size_t)ns;
            continue;
        } else if (act.kind == 3) { /* accept */
            free(stack);
            return PARSE_OK;
        } else {
            /* unreachable */
            if (err) { err->state = s; err->lookahead = la; }
            free(stack);
            return PARSE_ERR;
        }
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif
"""

    return "\n".join([header, prods_src, actions_src, gotos_src, names_src, runtime])
