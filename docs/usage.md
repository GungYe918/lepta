# lepta 사용 설명서 (Usage)

> **lepta**는 `.g` 문법 파일을 입력받아 **SLR(1)/LALR(1)** 파서 테이블을 생성하고, **Rust/C** 타깃으로 **단일 소스 파일(파서 런타임 + 테이블)** 을 방출하는 파서 제너레이터입니다. 이 문서는 설치 없이(리포지토리 기준) 바로 실행하여 문법을 만들고, 테이블을 점검하고, Rust/C 코드로 내보내 실제 프로그램에 통합하는 전 과정을 자세히 안내합니다. 초보자도 이 문서 하나로 간단한 **프로그래밍 언어** 수준의 파서를 만들 수 있도록 구성했습니다.

**핵심 요약**

* 입력: `.g`(lepta DSL)
* 내부: AST → BNF 전개 → 심볼/NULLABLE/FIRST/FOLLOW → (SLR|LALR) 테이블 → IR
* 출력: `parser.rs` 또는 `parser.c` **한 파일** (테이블+런타임 포함)
* 파싱법: **기본 LALR(1)**, 옵션으로 SLR(1)
* **신규 기능**(요약)

  * **유니코드 식별자 경계 인식 렉서**(가능하면 `regex` 모듈의 XID 사용)
  * **다문자 연산자 최장일치**(예: `<<`, `>>=`, `===` …) — 키워드 리터럴은 길이 내림차순 매칭
  * **패닉 모드 오류 복구**: `%recover panic;` + `%sync ...;` 로 구문 오류 후 진행 지속
  * (Rust) `--with-lexer` 시 **내장 단순 렉서**를 자동 포함(유니코드 경계 + 최장일치)

---

## 목차

1. [빠른 시작](#빠른-시작)
2. [CLI 개요와 설치 요건](#cli-개요와-설치-요건)
3. [lepta 문법(DSL) 작성 규칙](#lepta-문법dsl-작성-규칙)

   * [%token / %ignore / %start](#token--ignore--start)
   * [리터럴 키워드(다문자 연산자)](#리터럴-키워드다문자-연산자)
   * [우선순위와 결합성](#우선순위와-결합성)
   * [EBNF 연산자(?, \*, +)와 그룹](#ebnf-연산자-와-그룹)
   * [에러 복구 선언](#에러-복구-선언)
4. [BNF 전개·테이블 생성 파이프라인](#bnf-전개테이블-생성-파이프라인)
5. [CLI 사용법: check/build/lex](#cli-사용법-checkbuildlex)
6. [디버그 출력 읽는 법(상태/충돌/아이템)](#디버그-출력-읽는-법상태충돌아이템)
7. [Rust 타깃 통합 가이드](#rust-타깃-통합-가이드)

   * [내장 렉서 포함 모드](#내장-렉서-포함-모드)
   * [사용자 렉서/액션으로 계산기 만들기](#사용자-렉서액션으로-계산기-만들기)
   * [패닉 모드 동작 예시](#패닉-모드-동작-예시)
8. [C 타깃 통합 가이드](#c-타깃-통합-가이드)
9. [런타임 렉서 규약(파이썬/러스트)](#런타임-렉서-규약파이썬러스트)
10. [유니코드 지원 상세](#유니코드-지원-상세)
11. [레시피: 미니 언어 만들기](#레시피-미니-언어-만들기)
12. [문제 해결(FAQ)](#문제-해결faq)
13. [제한 사항과 로드맵](#제한-사항과-로드맵)

---

## 빠른 시작

### 1) 예제 문법: 사칙연산 + 단항 음수 + 거듭제곱 + 에러복구

```lepta
%token NUMBER /(?>0|[1-9][0-9_]*)(?:\.[0-9_]+)?(?:[eE][+\-]?[0-9_]+)?/;
%ignore /\s+/;
"(" : "("; ")" : ")"; "+":"+"; "-":"-"; "*":"*"; "/":"/"; "^":"^"; ";":";";

%use precedence cpp_core;     // + - < * / < ^ < (UMINUS)
%right UMINUS;                 // 단항 마이너스
%recover panic;                // 패닉 모드 복구 활성화
%sync ")", ";", "$";           // 동기화 토큰(닫는 괄호/세미콜론/EOF)

Expr  : Expr "+" Term
      | Expr "-" Term
      | Term
      ;

Term  : Term "*" Pow
      | Term "/" Pow
      | Pow
      ;

Pow   : Factor "^" Pow   // 우결합
      | Factor
      ;

Primary : NUMBER
        | "(" Expr ")"
        ;

Factor  : "-" Factor %prec UMINUS
        | Primary
        ;
```

### 2) 테이블 점검(기본 LALR)

```bash
python -m lepta.leptac check path/to/expr.g -D
```

* `-D`: AST/BNF 요약 + 상태 수 + 충돌 내역 + 상태 0 아이템을 출력합니다.

### 3) Rust 코드 방출 + 내장 렉서 포함

```bash
python -m lepta.leptac build path/to/expr.g \
  --parser lalr --lang rust --with-lexer -o out/parser.rs -D
```

* `--with-lexer`: **내장 단순 렉서**(유니코드 경계 + 최장일치 키워드)가 `parser.rs`에 함께 포함됩니다.

### 4) 간단 실행기(Rust) 붙이기

생성된 `parser.rs` 와 같은 디렉터리에 `main.rs` 를 두고 `mod parser;` 로 사용합니다. (샘플 코드는 [Rust 타깃 통합 가이드](#rust-타깃-통합-가이드) 참고)

---

## CLI 개요와 설치 요건

### 요구사항

* **Python 3.10+**
* (선택) `regex` 파이썬 모듈 — 설치되어 있으면 유니코드 식별자 경계(XID) 판정이 더 정확해집니다.

### 명령 형식

```
python -m lepta.leptac check FILE.g [--parser {lalr|slr}] [-D]
python -m lepta.leptac build FILE.g --lang {rust|c} -o OUT \
  [--parser {lalr|slr}] [--with-lexer] [-m MOD] [-D]
python -m lepta.leptac lex FILE.g --text "..." | --input input.txt
```

### 주요 옵션

* `check` : 문법을 **컴파일**하여 테이블을 생성하고 충돌 여부를 출력합니다.
* `build` : **코드 방출**(Rust/C). `--with-lexer` 로 Rust 내장 렉서를 포함.
* `lex`   : 문법을 이용해 입력 텍스트를 **토크나이즈**(디버깅용).
* `--parser`: `lalr`(기본) 또는 `slr` 선택.
* `--lang` : `rust` 또는 `c`.
* `-o`     : 출력 파일 경로.
* `-m`     : (Rust) 모듈명. 생략 시 출력 파일명에서 유도.
* `-D`     : 디버그 상세 출력.

---

## lepta 문법(DSL) 작성 규칙

### %token / %ignore / %start

* `%token NAME /regex/;` : 명명 토큰. 파서 테이블에서는 `NAME`이라는 단말 이름으로 사용됩니다.
* `%ignore /regex/;` : 렉서가 **스킵**할 패턴(공백, 주석 등). 여러 개 선언 가능.
* `%start IDENT;` : 시작 비단말 지정(선택). 생략 시 **첫 규칙의 좌변**이 시작 심볼.

> 파이썬 런타임 렉서/내장 러스트 렉서 모두 `%ignore`를 실제로 소비합니다.

### 리터럴 키워드(다문자 연산자)

리터럴은 아래처럼 **좌/우 동일 문자열**로 선언합니다.

```lepta
"+":"+"; "<<":"<<"; ">>=":">>="; "==":"=="; "<":"<"; "=":"=";
```

* **최장일치**: lepta 렉서는 키워드 목록을 **길이 내림차순**으로 매칭 → `>>=` 가 `>>`/`>`보다 먼저 잡힙니다.
* **유니코드 식별자 경계**: 키워드가 단어 성격(문자/숫자/언더스코어 포함)일 때, 앞/뒤가 식별자 문자가 아니어야 인식됩니다. 예: `if(` 는 `"if"` 키워드 OK, `iffy` 는 IDENT.

### 우선순위와 결합성

* 한 줄 선언: `%left`, `%right`, `%nonassoc`
* 아래로 갈수록 **강한 우선순위**

```lepta
%left "||";
%left "&&";
%nonassoc "==" "!=";
%left "<" "<=" ">" ">=";
%left "<<" ">>";
%left "+" "-";
%left "*" "/" "%";
%right UMINUS;         // 가상 라벨(단항 마이너스)
```

* 규칙 말미에 `%prec LABEL` 로 가상 라벨을 부여할 수 있습니다.

```lepta
Factor : "-" Factor %prec UMINUS | Primary ;
```

* 충돌 해소 규칙

  * 레벨 비교: 더 **강한 쪽**을 택함(shift vs reduce)
  * 같은 레벨: `left` → **reduce**, `right` → **shift**, `nonassoc` → **에러 칸**

> **템플릿**: `%use precedence cpp_core;` 같은 템플릿을 문법에서 사용할 수 있습니다(프로젝트에 정의되어 있을 때).

### EBNF 연산자 ( `?`, `*`, `+` )와 그룹

* `X?` → `ε | X`
* `X*` → `ε | X X*`
* `X+` → `X X*`
* `( ... )` 는 보조 비단말로 전개됩니다(내부 이름 `__grpN`, `__repN` 등 생성).

### 에러 복구 선언

```lepta
%recover panic;      // 패닉 모드 복구 활성화 (선언하지 않으면 페일-패스트)
%sync ")", ";", "$";  // 동기화 토큰 집합: 닫는 괄호/세미콜론/EOF
```

* 동작 요약(LR 루프)

  1. 에러 발생 시 입력을 **sync 토큰**이 나올 때까지 스킵
  2. 스택을 위에서부터 팝하며 `(state, lookahead)`에 **액션이 생길 때까지** 줄임
  3. 그래도 없으면 sync 토큰 **하나 소비** 후 계속
* Rust 런타임 상수: `RECOVER_PANIC`, `SYNC_TERMS`, `MAX_ERRORS` (생성물에 포함)

---

## BNF 전개·테이블 생성 파이프라인

1. `.g` 로드 → 2) DSL 파싱(AST) → 3) **EBNF→BNF** 전개 → 4) 심볼테이블 고정 → 5) NULLABLE/FIRST/FOLLOW 계산 → 6) **SLR(1)/LALR(1)** 테이블 생성 → 7) **precedence** 로 S/R 충돌 해소 → 8) IR 변환 → 9) 코드 방출(Rust/C)

* **SLR**: LR(0)+FOLLOW 기반 reduce → 간단하지만 보수적(충돌 가능성↑)
* **LALR**: canonical LR(1) 상태에서 **same-core 병합** (lookahead 전파) → 실전 친화, 충돌↓

---

## CLI 사용법: check/build/lex

### check

```bash
python -m lepta.leptac check examples/expr.g --parser lalr -D
```

* 결과: 상태 수, 충돌 수, 상태 0 아이템 일부.

### build (Rust)

```bash
python -m lepta.leptac build examples/expr.g \
  --parser lalr --lang rust --with-lexer -o out/parser.rs -D
```

* `--with-lexer`: **내장 러스트 렉서** 포함. 키워드 최장일치/유니코드 경계 지원.

### build (C)

```bash
python -m lepta.leptac build examples/expr.g --lang c -o out/parser.c -D
```

### lex (토크나이즈)

```bash
python -m lepta.leptac lex examples/expr.g --text "a<<b + 3"
# 또는
python -m lepta.leptac lex examples/expr.g --input sample.txt
```

* 출력 예)

  ```
  000: "<<"        '<<'  @1:2
  001: NUMBER      '3'    @1:7
  ```

---

## 디버그 출력 읽는 법(상태/충돌/아이템)

* `[Parsing Tables]` → `States: N`, `Conflicts: M`
* 충돌이 있으면 `state S, on SYMBOL: shift / reduce` 등으로 상세 내역 출력
* (LALR) 상태 아이템 형식: `[A -> α · β , {a,b}]` — lookahead 집합 표시

---

## Rust 타깃 통합 가이드

### 내장 렉서 포함 모드

`--with-lexer`를 사용하면 `parser.rs` 안에 다음이 자동 포함됩니다.

* `pub trait Lexer { fn next_kind(&mut self) -> Option<u16>; }`
* `pub struct SimpleLexer<'a>`: **키워드 최장일치 + 유니코드 경계 검사 + NUMBER 스캐너**
* `TERM_NAMES`, `NONTERM_NAMES`, `ACTION_KIND/ARG`, `GOTO_NEXT` 등 테이블 상수
* **패닉 모드 복구** 내장(`RECOVER_PANIC`, `SYNC_TERMS`, `MAX_ERRORS`)

**간단 실행기 예**

```rust
mod parser; // 생성된 parser.rs
use parser::{parse, Lexer, SimpleLexer};

fn main() {
    let mut lx = SimpleLexer::new("1 + 2 * 3; 4 + (5");
    match parse(&mut lx) {
        Ok(()) => println!("OK"),
        Err(e) => println!("ERR: state={} la={}", e.state, parser::TERM_NAMES[e.lookahead as usize]),
    }
}
```

### 사용자 렉서/액션으로 계산기 만들기

**값 생성 파서**는 `parse_with_actions` API를 제공합니다.

* `trait Lexer2 { fn next(&mut self) -> Option<Token>; }`
* `trait Actions { type Val; fn on_shift(&mut self, tok:&Token)->Val; fn on_reduce(&mut self, p:usize, lhs:u16, rhs:&[Val])->Val; }`

예) `NUMBER`를 `f64` 로 파싱하고 연산 수행(연산자 구조는 문법 우선순위로 확보됨).

> 구현 샘플은 프로젝트 예제(`example01_repl`) 또는 기존 대화의 `main.rs`/`ReprLexer`/`ReprActions` 참고.

### 패닉 모드 동작 예시

입력:

```
1 + * 2 ; 3 + 4
```

출력(예):

```
OK    | 1 + 2 * 3
OK    | 1 + * 2 ; 3 + 4
ERROR | 1 + (2 * 3
  → Parse error at state=20, lookahead=$, expected: ")", "+", "-"
...
```

* 첫 문장은 성공, 두 번째는 `;` 에서 동기화되어 계속 진행, 세 번째는 EOF에서 실패.

---

## C 타깃 통합 가이드

* 생성물: `parser.c` **단일 파일** (테이블 + 파서 루프 포함)
* 프로젝트에 포함 후, 사용자 쪽에서 `lexer_next_kind()`(또는 제공된 인터페이스)에 해당하는 함수를 연결해 주세요.
* (참고) 현재 C 경로는 러스트 경로와 동일한 IR을 사용하며, 테이블 조회는 O(1) dense 배열입니다.

---

## 런타임 렉서 규약(파이썬/러스트)

### 파이썬 런타임(Sample Lexer)

* **키워드 우선**: 길이 내림차순(최장일치)
* **유니코드 식별자 경계**: `regex` 모듈이 있으면 `\p{XID_Continue}` 기반, 없으면 `str.isalnum()/_` 근사
* **%token 정규식**: 후보 중 **가장 긴 매치**를 택함(동률 시 먼저 선언된 것)
* 산출 토큰의 `type`은 내부 심볼 이름과 **정확히 일치**해야 함 (예: `"+"`, `NUMBER`).

### 러스트 내장 렉서

* 키워드 배열이 **길이 내림차순**으로 방출되어 최장일치
* 경계 판정은 `char::is_alphanumeric()` + `_` 로 유니코드 경계 근사
* `NUMBER` 전용 스캐너가 포함(부호/소수/지수). 다른 정규식 토큰은 사용자가 확장 필요

---

## 유니코드 지원 상세

* **식별자 경계**: 유니코드 문자를 고려하여 키워드가 IDENT 내부에 섞여 매칭되지 않도록 보장
* 파이썬: `regex` 모듈 설치 시 `\p{XID_Continue}`로 엄밀 판정, 미설치 시 `isalnum/_` 근사
* 러스트: 표준 라이브러리의 유니코드 인식 `is_alphanumeric()` 기반
* **숫자/문자열 리터럴** 등은 `%token` 정규식으로 정의하세요. (예: 유니코드 이스케이프 등)

---

## 레시피: 미니 언어 만들기

### 1) 토큰들(식별자/숫자/연산자/키워드)

```lepta
%token IDENT /[A-Za-z_][A-Za-z0-9_]*/;
%token NUMBER /(?>0|[1-9][0-9_]*)(?:\.[0-9_]+)?(?:[eE][+\-]?[0-9_]+)?/;
%ignore /\s+/;
// 다문자 연산자 우선(최장일치)
"<<":"<<"; ">>":">>"; ">>=":">>="; "==":"=="; "!=":"!="; "&&":"&&"; "||":"||";
"+":"+"; "-":"-"; "*":"*"; "/":"/"; "%":"%"; "=":"="; ";":";";
"(":"("; ")":")"; "{":"{"; "}":"}";
```

### 2) 우선순위/결합성

```lepta
%left "||";
%left "&&";
%nonassoc "==" "!=";
%left "<<" ">>";
%left "+" "-";
%left "*" "/" "%";
%right UMINUS;
```

### 3) 표현식/문장/프로그램

```lepta
Expr
  : Expr "+" Expr | Expr "-" Expr
  | Expr "*" Expr | Expr "/" Expr | Expr "%" Expr
  | Expr "<<" Expr | Expr ">>" Expr
  | Expr "==" Expr | Expr "!=" Expr
  | "-" Expr %prec UMINUS
  | IDENT | NUMBER | "(" Expr ")"
  ;
Stmt : IDENT "=" Expr ";"
     | Expr ";"
     ;
Prog : Stmt* ;
```

* 실제 구현에서는 좌결합/우결합에 맞춰 `Expr→Term→Factor` 사다리 구조로 나누는 것이 상태 수/모호성 면에서 유리합니다.

### 4) 에러 복구 추천

```lepta
%recover panic;
%sync ";", "}", "$";
```

* 세미콜론/블록 닫힘/EOF에서 동기화 → 다음 문장으로 진행

---

## 문제 해결(FAQ)

**Q1. `%sync ";";` 를 썼는데 빌드 오류: `unknown %sync label ';'`**

* **리터럴 키워드 매핑**(`";":";";`)이 선언되어 있어야 `%sync`에서 사용할 수 있습니다. 토큰 집합은 **문법의 단말**만 허용합니다.

**Q2. `<<` 가 `<` + `<` 로 쪼개져 토큰화됩니다.**

* 키워드(리터럴) 목록이 **길이 내림차순(최장일치)** 로 정렬되어야 합니다. lepta의 파이썬 런타임/러스트 내장 렉서는 자동으로 그렇게 동작합니다. 문법에 `"<<":"<<";` 를 꼭 추가하세요.

**Q3. `ifx` 가 `if` 키워드 + `x` 로 잘못 쪼개집니다.**

* 유니코드 식별자 경계 검사 덕분에 기본적으로 방지됩니다. 문법에서 `"if":"if";` 를 키워드로 선언했는지 확인하세요. 런타임이 `regex` 모듈 없이 동작 중이라면 경계 판정이 근사치일 수 있습니다.

**Q4. S/R 충돌이 납니다.**

* LALR을 사용(`--parser lalr`)하고, 필요시 `%left/%right/%nonassoc` + `%prec` 를 추가하세요. 여전히 남으면 규칙 구조(사다리)로 분해하세요.

**Q5. 내장 렉서 말고 고급 렉서가 필요합니다.**

* Rust의 경우 생성된 `trait Lexer`/`Lexer2` 를 구현해 교체하세요. 파이썬 런타임은 `lepta.lex.SimpleLexer`를 참고해 커스텀 구현이 쉽습니다.

---

## 제한 사항과 로드맵

* (현행) 세만틱 액션은 **런타임 콜백**으로 제공(Rust: `parse_with_actions`), 문법 내 임베디드 액션 블록은 차기 계획
* (현행) C 경로는 러스트와 동일 IR 사용. 방출 템플릿 고도화 예정
* (현행) 문자열/주석/프리프로세서 등 복합 토큰은 `%token` 정규식 확장으로 처리 — 필요시 사용자 렉서 권장
* (로드맵) **PEG 백엔드**(f-string 등 좌측 팩토링이 어려운 문법에 대한 fallback)
* (로드맵) 테이블 압축/공유행 최적화, 에러 메시지 품질 향상(기대 집합 자동 복원, 범위 강조)

---

### 부록: 생성물(예—Rust) 내부 상수/테이블 개요

* `N_STATES`, `N_TERMS`, `N_NONTERMS`, `START_STATE`, `EOF_TERM`
* `PROD_LHS[]`, `PROD_RHS_LEN[]`
* `ACTION_KIND[]`, `ACTION_ARG[]` (dense 2D) — 조회 O(1)
* `GOTO_NEXT[]` (dense 2D)
* `TERM_NAMES[]`, `NONTERM_NAMES[]`
* (옵션) `RECOVER_PANIC`, `SYNC_TERMS`, `MAX_ERRORS`(패닉 모드)

> 위 상수/배열은 디버깅 시 상태/토큰 이름 복원과 기대 집합 계산에 유용합니다.
