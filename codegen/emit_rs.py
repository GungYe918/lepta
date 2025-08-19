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

def _fmt_peg_buckets_wrapped(bucket_lists, per=16) -> str:
    """
    bucket_lists: 길이 256의 리스트. 각 원소는 u16 인덱스 리스트 (예: [0,1]) 또는 빈 리스트.
    per: 한 줄에 출력할 엔트리 개수 (기본 16)
    반환: Rust 배열 리터럴 내부에 넣을 문자열 (주석으로 0x범위 표기)
    """
    entries = []
    for ids in bucket_lists:
        if ids:
            entries.append("&[" + ", ".join(str(i) for i in ids) + "]")
        else:
            entries.append("&[]")

    lines = []
    for i in range(0, 256, per):
        chunk = ", ".join(entries[i:i+per])
        lines.append(f"/* 0x{i:02X}..0x{i+per-1:02X} */ {chunk}")
    return ",\n".join(lines)



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

def _escape_rs_char(ch: str) -> str:
    """Rust 단일 문자 리터럴용 이스케이프."""
    if ch == "'":
        return "\\'"
    if ch == "\\":
        return "\\\\"
    # 일반 ASCII 제어문자들 처리(필요 최소)
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    return ch

def _peg_parse_min(src: str):
    """
    매우 작은 PEG 파서(코드생성 전용).
    지원 요소: Literal('..'), CharClass([...]), Any(.), Seq, Choice(/),
              Repeat(* + ?), Group(()), Not(!), Ref(IDENT)
    문법:  Rule <- expr
    반환: {"rules": dict[name -> node]}
    node: dict 형태로 k 키로 구분
      - {"k":"lit","s":str}
      - {"k":"any"}
      - {"k":"class","neg":bool,"items":[("ch",ch)|("range",a,b), ...]}
      - {"k":"seq","items":[node,...]}
      - {"k":"choice","alts":[node,...]}
      - {"k":"rep","op":"*|+|?","node":node}
      - {"k":"not","node":node}
      - {"k":"ref","name":str}
    """
    i, n = 0, len(src)

    def peek():
        return src[i] if i < n else ""

    def eat():
        nonlocal i
        ch = src[i] if i < n else ""
        i += 1
        return ch

    def skip_ws():
        nonlocal i
        while i < n and src[i] in " \t\r\n":
            i += 1

    def parse_ident():
        nonlocal i
        skip_ws()
        j = i
        if i < n and (src[i].isalpha() or src[i] == "_"):
            i += 1
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            return src[j:i]
        return None

    def parse_lit():
        skip_ws()
        if peek() != "'":
            return None
        eat()  # '
        out = []
        while True:
            ch = eat()
            if ch == "":
                raise SyntaxError("Unterminated literal in %peg block")
            if ch == "\\":
                # 간단 이스케이프 처리 (\' \\)
                esc = eat()
                if esc == "":
                    raise SyntaxError("Bad escape in literal")
                if esc == "n":
                    out.append("\n")
                elif esc == "r":
                    out.append("\r")
                elif esc == "t":
                    out.append("\t")
                else:
                    out.append(esc)
                continue
            if ch == "'":
                break
            out.append(ch)
        return {"k": "lit", "s": "".join(out)}

    def parse_class():
        skip_ws()
        if peek() != "[":
            return None
        eat()  # [
        neg = False
        if peek() == "^":
            eat()
            neg = True
        items = []
        def read_char():
            ch = eat()
            if ch == "":
                raise SyntaxError("Unterminated char class")
            if ch == "\\":
                esc = eat()
                if esc == "":
                    raise SyntaxError("Bad escape in char class")
                if esc == "n":
                    return "\n"
                if esc == "r":
                    return "\r"
                if esc == "t":
                    return "\t"
                return esc
            return ch
        while True:
            if peek() == "]":
                eat()
                break
            a = read_char()
            if peek() == "-" and src[i+1:i+2] not in ("", "]"):
                eat()  # '-'
                b = read_char()
                items.append(("range", a, b))
            else:
                items.append(("ch", a))
        return {"k": "class", "neg": neg, "items": items}

    def parse_any():
        skip_ws()
        if peek() == ".":
            eat()
            return {"k": "any"}
        return None

    def parse_primary():
        skip_ws()
        if peek() == "(":
            eat()
            e = parse_expr()
            skip_ws()
            if peek() != ")":
                raise SyntaxError("Expected ')' in %peg group")
            eat()
            return e
        if peek() == "!":
            eat()
            sub = parse_primary()
            return {"k": "not", "node": sub}
        lit = parse_lit()
        if lit:
            return lit
        cls = parse_class()
        if cls:
            return cls
        anyx = parse_any()
        if anyx:
            return anyx
        ident = parse_ident()
        if ident:
            return {"k": "ref", "name": ident}
        return None

    def parse_suffix(node):
        skip_ws()
        if peek() in ("*", "+", "?"):
            op = eat()
            return {"k": "rep", "op": op, "node": node}
        return node

    def parse_seq():
        nonlocal i
        skip_ws()
        items = []
        while True:
            save = i
            if peek() in (")", "/", ""):
                break
            p = parse_primary()
            if not p:
                i = save
                break
            items.append(parse_suffix(p))
            skip_ws()
            if peek() in (")", "/", ""):
                break
        if not items:
            # 빈 시퀀스는 ε 허용(여기선 {'k':'seq','items':[]}로 표기)
            return {"k": "seq", "items": []}
        return {"k": "seq", "items": items}

    def parse_expr():
        left = parse_seq()
        skip_ws()
        alts = [left]
        while peek() == "/":
            eat()
            alts.append(parse_seq())
            skip_ws()
        if len(alts) == 1:
            return left
        return {"k": "choice", "alts": alts}

    # Rules
    rules = {}
    while True:
        skip_ws()
        if i >= n:
            break
        name = parse_ident()
        if not name:
            # 빈 줄 등 스킵
            if i < n:
                # 예외: 주석 등은 미지원. 입력이 남았는데 식별자 아니면 에러.
                raise SyntaxError("Expected rule name in %peg block")
            break
        skip_ws()
        # '<-'
        if not (peek() == "<" and src[i+1:i+2] == "-"):
            raise SyntaxError("Expected '<-' after rule name in %peg block")
        i += 2
        expr = parse_expr()
        rules[name] = expr
        skip_ws()
    return {"rules": rules}


def _analyze_peg_blocks(parsed_blocks: dict[str, dict]):
    """
    각 %peg 규칙에 대해 다음 정보를 정적 분석해 반환한다.
      - trig : 가능한 '선두 바이트(UTF-8)' 집합 (FIRST1 근사)
      - min  : 소비 바이트의 보수적 최소 길이
      - max  : 소비 바이트의 보수적 최대 길이 (0xFFFFFFFF = 무한대)
    반환 스키마: { block: { rule: {"trig": set[int], "min": int, "max": int} } }
    """
    INF = 0xFFFFFFFF
    from functools import lru_cache

    # --- 원자 노드 정보 ---

    def lit_info(s: str):
        b = s.encode("utf-8")
        if not b:
            return set(), 0, 0, True  # ε
        return {b[0]}, len(b), len(b), False

    def class_info(node):
        trig = set()
        for kind, *rest in node["items"]:
            if kind == "ch":
                b = rest[0].encode("utf-8")
                if b:
                    trig.add(b[0])
            elif kind == "range":
                a, z = rest
                ao, zo = ord(a), ord(z)
                # 안전하게 ASCII 범위만 확장
                if ao <= 0x7F and zo <= 0x7F:
                    for v in range(ao, zo + 1):
                        trig.add(v & 0xFF)
        # UTF-8 코드포인트 1..4 바이트
        return trig, 1, 4, False

    def any_info():
        return set(), 1, 4, False

    # --- 유틸 ---

    def sum_len(a, b):
        if a == INF or b == INF:
            return INF
        return a + b

    def max_choice(a, b):
        # choice(합집합)의 최댓값: 대안들 중 최댓값 (무한 전파)
        if a == INF or b == INF:
            return INF
        return max(a, b)

    # 규칙 참조용 캐시
    @lru_cache(None)
    def info_of(block: str, rule: str):
        expr = parsed_blocks[block]["rules"][rule]
        return info_expr(block, expr)

    # --- 재귀 본체 ---

    def info_expr(block: str, node: dict):
        k = node["k"]

        if k == "lit":
            return lit_info(node["s"])

        if k == "class":
            return class_info(node)

        if k == "any":
            return any_info()

        if k == "ref":
            # (서클) 재귀 발견 시 보수적으로 무한 상한/nullable 처리
            try:
                return info_of(block, node["name"])
            except RecursionError:
                return set(), 0, INF, True

        if k == "not":
            # 부정 전방 탐색은 소비하지 않음(lookahead)
            return set(), 0, 0, True

        if k == "rep":
            op = node["op"]
            t, mn, mx, eps = info_expr(block, node["node"])
            if op == "*":
                # 0회 이상: FIRST=t, min=0, max=INF, ε 가능
                return set(t), 0, INF, True
            if op == "+":
                # 1회 이상: min=mn(노드가 ε이면 0), max=INF, ε 여부는 노드와 동일
                return set(t), mn, INF, eps
            if op == "?":
                # 0/1회: FIRST=t, min=0, max=mx, ε 가능
                return set(t), 0, mx, True
            # 알 수 없는 접미사 → 보수적
            return set(), 0, INF, True

        if k == "seq":
            items = node.get("items", [])
            if not items:
                return set(), 0, 0, True
            first = set()
            min_total = 0
            max_total = 0
            all_nullable = True
            for it in items:
                t, mn, mx, eps = info_expr(block, it)
                # FIRST(시퀀스): 앞에서부터 최초 비널까지 FIRST 합집합
                if all_nullable:
                    first |= set(t)
                # 최소/최대 길이는 '합'(시퀀스)이어야 함
                min_total = sum_len(min_total, mn)
                max_total = sum_len(max_total, mx)  # ★ 기존 버그: max를 '합'이 아니라 '최대값'으로 계산하던 문제 수정
                if not eps:
                    all_nullable = False
            return first, min_total, max_total, all_nullable

        if k == "choice":
            alts = node.get("alts", [])
            if not alts:
                return set(), 0, 0, True
            first = set()
            min_any = 0x7FFFFFFF
            max_any = 0
            any_eps = False
            for a in alts:
                t, mn, mx, eps = info_expr(block, a)
                first |= set(t)
                min_any = min(min_any, mn)
                max_any = max_choice(max_any, mx)
                any_eps = any_eps or eps
            if min_any == 0x7FFFFFFF:
                min_any = 0
            return first, min_any, max_any, any_eps

        # 알 수 없는 노드 → 보수적
        return set(), 0, INF, True

    # --- 블록/규칙 단위 결과 묶기 ---
    out = {}
    for bname, parsed in parsed_blocks.items():
        rules = parsed["rules"]
        out[bname] = {}
        for rname in rules.keys():
            t, mn, mx, _eps = info_of(bname, rname)
            out[bname][rname] = {"trig": set(t), "min": int(mn), "max": int(mx)}
    return out



def _peg_emit_rust_from_blocks(ast) -> str:
    """
    Grammar AST의 %peg 블록과 %token ... %peg(Block.Rule)들을 사용하여,
    러스트 측에서 **백트래킹 가능한** PEG 런타임(전용 함수들)을 생성한다.

    - 각 Block.Rule에 대해: fn peg_<Block>_<Rule>(s:&str, pos:usize)->Option<usize>
    - 필요 헬퍼:
        next_char, eat_lit, (클래스 검사 inline)
    - PEG 토큰 테이블: PEG_SPECS: &[PegSpec] (name, trigger, func)
    - PegSpec에 trig(선두 바이트 집합), min_len/max_len 추가
    - 256개 버킷(선두 바이트) + fallback 후보 테이블 생성
    - 리터럴 매치를 바이트 기반 starts_with로 고정(미세 최적화)

    반환: 러스트 코드 문자열(모듈 스코프에 그대로 삽입됨)
    """

    # 1) 사용되는 (term, block, rule, trigger) 수집
    peg_tokens = []
    for pt in getattr(ast, "decl_peg_tokens", []):
        b, r = pt.peg_ref
        trig = getattr(pt, "trigger", None) or None
        peg_tokens.append((pt.name, b, r, trig))

    if not peg_tokens:
        return ""  # PEG 미사용 시 빈 문자열

    # 2) 블록 소스 맵
    block_src: dict[str, str] = {pb.name: pb.src for pb in getattr(ast, "decl_peg_blocks", [])}

    # 3) 파싱
    parsed_blocks: dict[str, dict] = {}
    for _term, b, _r, _tr in peg_tokens:
        if b not in block_src:
            raise SyntaxError(f"emit_rs: %peg block '{b}' not found for PEG token")
        if b not in parsed_blocks:
            parsed_blocks[b] = _peg_parse_min(block_src[b])

    # 4) 정적 분석(FIRST1/길이)
    analysis = _analyze_peg_blocks(parsed_blocks)

    # 5) 코드 제너레이터: 노드 → 러스트 Option<usize> 표현 생성 (lit은 바이트기반)
    tmp_id = {"n": 0}
    def new_tmp(prefix="p"):
        tmp_id["n"] += 1
        return f"{prefix}{tmp_id['n']}"

    def _escape_rs(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def emit_class_pred(cls):
        conds = []
        for kind, *rest in cls["items"]:
            if kind == "ch":
                ch = rest[0]
                if ch == "'": ch = "\\'"
                if ch == "\\": ch = "\\\\"
                if ch == "\n": ch = "\\n"
                if ch == "\r": ch = "\\r"
                if ch == "\t": ch = "\\t"
                conds.append(f"ch == '{ch}'")
            elif kind == "range":
                a, b = rest
                if a == "'": a = "\\'"
                if a == "\\": a = "\\\\"
                if a == "\n": a = "\\n"
                if a == "\r": a = "\\r"
                if a == "\t": a = "\\t"
                if b == "'": b = "\\'"
                if b == "\\": b = "\\\\"
                if b == "\n": b = "\\n"
                if b == "\r": b = "\\r"
                if b == "\t": b = "\\t"
                conds.append(f"('{a}' <= ch && ch <= '{b}')")
        in_set = " || ".join(conds) if conds else "false"
        return f"!({in_set})" if cls["neg"] else f"({in_set})"

    def emit_expr(block, node, pos_expr: str) -> str:
        k = node["k"]

        if k == "lit":
            lit = _escape_rs(node["s"])
            return (
                "{ "
                f"let bytes = s.as_bytes(); "
                f"let lit = \"{lit}\".as_bytes(); "
                f"if bytes.len() >= {pos_expr} + lit.len() && "
                f"   &bytes[{pos_expr}..{pos_expr}+lit.len()] == lit "
                f"{{ Some({pos_expr} + lit.len()) }} else {{ None }} "
                "}"
            )

        if k == "any":
            return (
                "{ "
                f"let mut iter = s[{pos_expr}..].chars(); "
                "if let Some(ch) = iter.next() { "
                "let adv = ch.len_utf8(); "
                f"Some({pos_expr} + adv) "
                "} else { None } "
                "}"
            )

        if k == "class":
            pred = emit_class_pred(node)
            return (
                "{ "
                f"let mut iter = s[{pos_expr}..].chars(); "
                "if let Some(ch) = iter.next() { "
                f"if {pred} {{ Some({pos_expr} + ch.len_utf8()) }} else {{ None }} "
                "} else { None } "
                "}"
            )

        if k == "ref":
            fname = f"peg_{block}_{node['name']}"
            return f"{fname}(s, {pos_expr})"

        if k == "not":
            inner = emit_expr(block, node["node"], pos_expr)
            return (
                "{ "
                f"if {inner}.is_some() {{ None }} else {{ Some({pos_expr}) }} "
                "}"
            )

        if k == "rep":
            op = node["op"]
            inner_cl = new_tmp("e")
            inner_code = emit_expr(block, node["node"], inner_cl)

            if op == "*":
                p = new_tmp("p")
                pprev = new_tmp("pp")
                return (
                    "{ "
                    f"let mut {p} = {pos_expr}; "
                    "loop { "
                    f"let {pprev} = {p}; "
                    f"let {inner_cl} = {p}; "
                    f"match {inner_code} {{ "
                    f"    Some(npos) => {{ {p} = npos; if {p} == {pprev} {{ break; }} }} "
                    "    None => break "
                    "} } "
                    f"Some({p}) "
                    "}"
                )

            if op == "+":
                p = new_tmp("p")
                pprev = new_tmp("pp")
                first = new_tmp("pf")
                return (
                    "{ "
                    f"let mut {p} = {pos_expr}; "
                    f"let {first} = {{ "
                    f"    let {inner_cl} = {p}; "
                    f"    match {inner_code} {{ Some(npos) => npos, None => return None }} "
                    "}; "
                    f"{p} = {first}; "
                    "loop { "
                    f"    let {pprev} = {p}; "
                    f"    let {inner_cl} = {p}; "
                    f"    match {inner_code} {{ "
                    f"        Some(npos) => {{ {p} = npos; if {p} == {pprev} {{ break; }} }} "
                    "        None => break "
                    "    } } "
                    f"Some({p}) "
                    "}"
                )

            if op == "?":
                tmp = new_tmp("p")
                return (
                    "{ "
                    f"let {tmp} = {pos_expr}; "
                    f"let res = {emit_expr(block, node['node'], tmp)}; "
                    f"match res {{ Some(npos) => Some(npos), None => Some({pos_expr}) }} "
                    "}"
                )

            raise ValueError("unknown rep op")

        if k == "seq":
            items = node.get("items", [])
            if not items:
                return f"Some({pos_expr})"
            cur = new_tmp("p")
            parts = [f"{{ let mut {cur} = {pos_expr}; "]
            for it in items:
                scratch = new_tmp("pp")
                expr_code = emit_expr(block, it, scratch)
                parts.append(f"let {scratch} = {cur}; ")
                parts.append(f"match {expr_code} {{ Some(npos) => {{ {cur} = npos; }}, None => return None }} ")
            parts.append(f"Some({cur}) }}")
            return "".join(parts)

        if k == "choice":
            # ★ 핵심 수정: 분기 안에서 `return Some(...)`로 함수 전체를 조기 종료하지 말고,
            #   블록 표현식으로 Option을 구성해 상위 문맥으로 반환하게 만든다.
            alts = node["alts"]
            parts = ["{ "]
            for i, a in enumerate(alts):
                expr = emit_expr(block, a, pos_expr)
                if i == 0:
                    parts.append(f"if let Some(npos) = {expr} {{ Some(npos) }} ")
                else:
                    parts.append(f"else if let Some(npos) = {expr} {{ Some(npos) }} ")
            parts.append("else { None } }")
            return "".join(parts)

        raise ValueError(f"unknown PEG node kind: {k}")

    # 6) 규칙 함수 생성
    rust_funcs: list[str] = []
    for bname, parsed in parsed_blocks.items():
        rules = parsed["rules"]
        for rname, expr in rules.items():
            body = emit_expr(bname, expr, "pos")
            fn = (
                f"#[inline(always)]\n"
                f"#[allow(unused_parens)]\n"
                f"fn peg_{bname}_{rname}(s: &str, pos: usize) -> Option<usize> {{ "
                f"{body} "
                "}\n"
            )
            rust_funcs.append(fn)

    # 7) PegSpec + PEG_SPECS + 버킷 생성
    specs_rows = []
    trig_lists = []
    min_list = []
    max_list = []
    for idx, (term, b, r, trig_char) in enumerate(peg_tokens):
        trig_rs = "None" if not trig_char else f"Some('{_escape_rs_char(trig_char)}')"
        info = analysis.get(b, {}).get(r, {"trig": set(), "min": 0, "max": 0xFFFFFFFF})

        trig = set(int(v) & 0xFF for v in info["trig"])
        if trig_char:
            tb = trig_char.encode('utf-8')
            if tb:
                trig.add(tb[0])

        trig = sorted(trig)
        trig_lists.append(trig)

        min_v = int(info["min"])
        max_v = int(info["max"])
        min_list.append(min_v)
        max_list.append(max_v)

        trig_arr = ", ".join(str(v) for v in trig)
        specs_rows.append(
            "PegSpec { "
            f"name: \"{_escape_rs(term)}\", trigger: {trig_rs}, func: peg_{b}_{r}, "
            f"trig: &[{trig_arr}], min_len: {min_v}u32, max_len: {max_v}u32 "
            "}"
        )

    specs_src = ",\n    ".join(specs_rows)

    buckets = [[] for _ in range(256)]
    fallback = []
    for i, trig in enumerate(trig_lists):
        if trig:
            for b in trig:
                if 0 <= b < 256:
                    buckets[b].append(i)
        else:
            fallback.append(i)

    def fmt_u16_arr(xs):
        return ", ".join(str(int(x)) for x in xs)

    buckets_src = _fmt_peg_buckets_wrapped(buckets, per=16)
    fallback_src = fmt_u16_arr(fallback)

    helpers = (
        "/// ---- PEG 런타임(토큰용, 최적화판) ----\n"
        "/// - 백트래킹/선택/반복/부정전방탐색 지원\n"
        "/// - 버킷 디스패치(선두 바이트) + 길이 기반 조기 건너뛰기\n"
        "const PEG_INF: u32 = 0xFFFF_FFFF;\n"
        "#[derive(Copy, Clone)]\n"
        "struct PegSpec {\n"
        "    name: &'static str,\n"
        "    trigger: Option<char>,\n"
        "    func: fn(&str, usize) -> Option<usize>,\n"
        "    trig: &'static [u8],        // 가능한 선두 바이트 집합(비어있으면 fallback)\n"
        "    min_len: u32,               // 보수적 최소 소비 바이트\n"
        "    max_len: u32,               // 보수적 최대 소비 바이트(PEG_INF=무한)\n"
        "}\n\n"
        "static PEG_SPECS: &[PegSpec] = &[\n"
        f"    {specs_src}\n"
        "];\n\n"
        "/// 0..255 선두 바이트 버킷\n"
        "static PEG_BUCKETS: [&[u16]; 256] = [\n"
        f"{buckets_src}\n"
        "];\n"
        "/// trig가 비어있거나 선두를 알 수 없는 규칙들\n"
        f"static PEG_FALLBACK: &[u16] = &[{fallback_src}];\n\n"
    )
    return helpers + "".join(rust_funcs)


def _lexer_src_from_ast(ast) -> str:
    """
    Rust 내장 렉서 방출(PEG 통합판)
    =================================
    - 키워드(리터럴) 최장일치 + 단어경계
    - %ignore /\\s+/ 스킵
    - %token NUMBER (하드코딩) 스캔
    - %peg 토큰: 버킷 디스패치 + min/max 길이 기반 조기 건너뛰기
    - %token ... %peg(Block.Rule) 및 RHS @peg(...) (로워링 후 전역 PEG 토큰) 지원
      * 토큰 단위 최장일치(여러 PEG 토큰 후보 중 가장 긴 것 선택, 동률이면 선언순)
      * [trigger='x'] 지정 시 시작 문자가 x일 때만 시도
    """
    # --- 키워드: 중복 제거 + 길이 내림차순 ---
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
    kws_arr = ", ".join(f"\"{_escape_rs(k)}\"" for k in kws)

    # ignore: /\\s+/ 존재 여부
    has_ws_ignore = any(ig.pattern == r"\s+" for ig in ast.decl_ignores)

    # token 이름 수집
    token_names = [td.name for td in ast.decl_tokens]
    has_number = "NUMBER" in token_names
    has_ident  = "IDENT"  in token_names

    # PEG 섹션(필요 시) 생성
    peg_section_src = _peg_emit_rust_from_blocks(ast)
    has_peg = bool(peg_section_src.strip())

    # 1) skip_ignores 본문
    if has_ws_ignore:
        skip_ignores_code = (
            "while let Some(ch) = self.peek_char() { "
            "if ch.is_whitespace() { self.take_while(|c| c.is_whitespace()); } else { break; } }"
        )
    else:
        skip_ignores_code = "/* no %ignore /\\s+/; */"

    # 2) NUMBER: (하네스 기준) 선행 부호는 파서가 처리, 렉서는 **숫자 시작일 때만** 인식
    if has_number:
        match_number_code = """
        // 자연수/정수(요구 조건: 테스트에서는 자연수만 사용)
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
            } else { return None; }
        }
        // 선택적 소수/지수는 문법상 허용될 수 있으나, 테스트는 자연수만 사용
        if self.starts_with_at(".") {
            self.advance_by(1);
            let n = self.take_while(|c| c.is_ascii_digit() || c == '_');
            if n == 0 { return None; }
        }
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

    looks_like_helper = ""
    if has_number:
        looks_like_helper = """
    #[inline]
    fn looks_like_number_start(&self) -> bool {
        if self.i >= self.text.len() { return false; }
        if let Some(c0) = self.text[self.i..].chars().next() {
            return c0.is_ascii_digit(); // 선행 부호는 여기서 처리하지 않음
        }
        false
    }
""".rstrip("\n")

    # NUMBER 매처 함수 본체 삽입
    number_helpers = ""
    if has_number:
        number_helpers = f"""
    fn match_number(&mut self) -> Option<&'static str> {{
{match_number_code}
    }}
""".rstrip("\n")

    # 3) IDENT (하네스 동일한 근사: 시작은 알파벳/_; 이어붙임은 \\w 근사)
    ident_helpers = ""
    if has_ident:
        ident_helpers = """
    #[inline]
    fn looks_like_ident_start(&self) -> bool {
        matches!(self.peek_char(), Some(ch) if ch.is_alphabetic() || ch == '_')
    }
    fn match_ident(&mut self) -> Option<&'static str> {
        if !self.looks_like_ident_start() { return None; }
        if let Some(ch) = self.peek_char() {
            if ch.is_alphabetic() || ch == '_' {
                self.advance_by(ch.len_utf8());
            } else { return None; }
        } else { return None; }
        self.take_while(|c| c.is_alphanumeric() || c == '_');
        Some("IDENT")
    }
""".rstrip("\n")

    num_first_try = ""
    if has_number:
        num_first_try = (
            "if self.looks_like_number_start() {\n"
            "            if let Some(tok) = self.match_number() {\n"
            "                for (i, name) in TERM_NAMES.iter().enumerate() {\n"
            "                    if *name == tok { return Some(i as u16); }\n"
            "                }\n"
            "                panic!(\"token not found in TERM_NAMES: {}\", tok);\n"
            "            }\n"
            "        }"
        )

    num_fallback = ""
    if has_number:
        num_fallback = (
            "if let Some(tok) = self.match_number() {\n"
            "            for (i, name) in TERM_NAMES.iter().enumerate() {\n"
            "                if *name == tok { return Some(i as u16); }\n"
            "            }\n"
            "            panic!(\"token not found in TERM_NAMES: {}\", tok);\n"
            "        }"
        )

    ident_try = ""
    if has_ident:
        ident_try = (
            "if let Some(tok) = self.match_ident() {\n"
            "            for (i, name) in TERM_NAMES.iter().enumerate() {\n"
            "                if *name == tok { return Some(i as u16); }\n"
            "            }\n"
            "            panic!(\"token not found in TERM_NAMES: {}\", tok);\n"
            "        }"
        )

    # match_peg 구현: PEG 블록이 없으면 스텁 생성(PEG_SPECS 미정의 에러 예방)
    if has_peg:
        match_peg_src = """
    /// PEG 토큰: 버킷 디스패치 + min/max 기반 조기 건너뛰기 + 최장일치
    fn match_peg(&mut self) -> Option<(&'static str, usize)> {
        if PEG_SPECS.is_empty() { return None; }
        let s = self.text;
        let start = self.i;
        if start >= s.len() { return None; }

        let bytes = s.as_bytes();
        let first = bytes[start] as usize;

        let mut best_len: usize = 0;
        let mut best_name: Option<&'static str> = None;

        #[inline(always)]
        fn check_bucket(
            bucket: &[u16], s: &str, start: usize,
            best_len: &mut usize, best_name: &mut Option<&'static str>,
        ) {
            for &idx in bucket {
                let spec = &PEG_SPECS[idx as usize];

                // trigger 검사
                if let Some(tr) = spec.trigger {
                    if let Some(ch) = s[start..].chars().next() {
                        if ch != tr { continue; }
                    } else { continue; }
                }

                // 길이 하한/상한 조기 가지치기
                let remain = (s.len() - start) as u32;
                if remain < spec.min_len { continue; }
                if *best_len > 0 {
                    if spec.max_len != PEG_INF && (spec.max_len as usize) <= *best_len {
                        continue;
                    }
                }

                if let Some(end_pos) = (spec.func)(s, start) {
                    if end_pos > start {
                        let len = end_pos - start;
                        if len > *best_len {
                            *best_len = len;
                            *best_name = Some(spec.name);
                        }
                    }
                }
            }
        }

        // 1) 선두 바이트 버킷
        check_bucket(PEG_BUCKETS[first], s, start, &mut best_len, &mut best_name);
        // 2) fallback 버킷
        if best_len == 0 {
            check_bucket(PEG_FALLBACK, s, start, &mut best_len, &mut best_name);
        }

        match best_name {
            Some(nm) if best_len > 0 => Some((nm, best_len)),
            _ => None
        }
    }
"""
    else:
        match_peg_src = """
    #[inline]
    fn match_peg(&mut self) -> Option<(&'static str, usize)> { None }
"""

    # 최종 Rust 소스(핵심 변경: next_kind에서 **PEG 먼저 시도**, 그 다음 키워드/숫자/식별자 순)
    return f"""
// ====== Generated Simple Lexer (PEG-enabled, optimized) ======
// 제한: 공백 스킵(\\s+), 키워드 리터럴(최장일치), NUMBER, PEG 토큰(@peg 포함) 지원
//  - PEG: 버킷 디스패치(선두 바이트) + 길이 기반 조기 건너뛰기
//  - 기타 정규식 토큰은 내장 렉서가 지원하지 않습니다.

{peg_section_src}

pub struct SimpleLexer<'a> {{
    text: &'a str,
    i: usize,
}}

impl<'a> SimpleLexer<'a> {{
    pub fn new(input: &'a str) -> Self {{ Self {{ text: input, i: 0 }} }}

    fn at_eof(&self) -> bool {{ self.i >= self.text.len() }}
    fn peek_char(&self) -> Option<char> {{ self.text[self.i..].chars().next() }}
    fn advance_by(&mut self, n_bytes: usize) {{ self.i += n_bytes; }}
    fn starts_with_at(&self, s: &str) -> bool {{ self.text[self.i..].as_bytes().starts_with(s.as_bytes()) }}

    fn take_while<F: FnMut(char)->bool>(&mut self, mut f: F) -> usize {{
        let mut n = 0usize;
        for ch in self.text[self.i..].chars() {{
            if f(ch) {{ n += ch.len_utf8(); }} else {{ break; }}
        }}
        self.advance_by(n);
        n
    }}

    #[inline]
    fn is_word_kw(s: &str) -> bool {{
        s.chars().any(|c| c.is_alphanumeric() || c == '_')
    }}

{looks_like_helper}
{number_helpers}
{ident_helpers}

    fn match_keyword(&mut self) -> Option<&'static str> {{
        const KEYWORDS: &[&str] = &[{kws_arr}]; // 길이 내림차순
        for &kw in KEYWORDS {{
            if self.starts_with_at(kw) {{
                if Self::is_word_kw(kw) {{
                    // 단어 경계 검사
                    let mut prev_ok = true;
                    if self.i > 0 {{
                        if let Some(prev) = self.text[..self.i].chars().rev().next() {{
                            if prev.is_alphanumeric() || prev == '_' {{ prev_ok = false; }}
                        }}
                    }}
                    let after = self.i + kw.len();
                    let mut next_ok = true;
                    if after < self.text.len() {{
                        if let Some(next) = self.text[after..].chars().next() {{
                            if next.is_alphanumeric() || next == '_' {{ next_ok = false; }}
                        }}
                    }}
                    if !(prev_ok && next_ok) {{ continue; }}
                }}
                self.advance_by(kw.len());
                return Some(kw);
            }}
        }}
        None
    }}

    fn skip_ignores(&mut self) {{ {skip_ignores_code} }}

{match_peg_src}
}}

impl<'a> Lexer for SimpleLexer<'a> {{
    fn next_kind(&mut self) -> Option<u16> {{
        self.skip_ignores();
        if self.at_eof() {{ return Some(EOF_TERM); }}

        // 1) PEG 토큰을 최우선으로 시도 (f\\\"...\\\" / \\\"...\\\")
        if let Some((nm, len)) = self.match_peg() {{
            self.advance_by(len);
            for (i, name) in TERM_NAMES.iter().enumerate() {{
                if *name == nm {{ return Some(i as u16); }}
            }}
            panic!("PEG token not found in TERM_NAMES: {{}}", nm);
        }}

        // 2) 키워드
        if let Some(kw) = self.match_keyword() {{
            for (i, name) in TERM_NAMES.iter().enumerate() {{
                if *name == kw {{ return Some(i as u16); }}
            }}
            panic!("keyword not found in TERM_NAMES: {{}}", kw);
        }}

        // 3) 숫자 (선두가 숫자일 때만 시도)
        {num_first_try}

        // 4) IDENT
        {ident_try}

        // 5) 숫자 폴백(위 조건들에서 잡히지 않았지만 숫자인 경우)
        {num_fallback}

        // 6) 실패
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
