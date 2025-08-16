# lepta 사용 설명서 (Usage)

> **lepta**는 `.g` 문법 파일을 입력받아 **(LALR/SLR) 파서 테이블**을 생성하고, **Rust/C** 타깃의 단일 소스 코드로 **파서 런타임 + 테이블**을 방출하는 파서 제너레이터입니다. 본 문서는 초보자도 이 문서 하나로 Rasca를 설치 없이(리포지토리 기준) 실행, 문법을 작성, 테이블을 검사하고, Rust/C 코드로 방출해 실행할 수 있도록 **자세한 절차와 예제**를 제공합니다.

* 최종 목표: `.g` → **테이블** → **파서 코드(.rs / .c)** → **응용 프로그램**
* 파싱 기법: **기본 LALR(1)**, 옵션으로 **SLR(1)** 지원
* 산출물: 단일 파일(레몬(lemon) 스타일) – 테이블과 파서 루프, 심볼 이름 등이 모두 포함

---

## 목차

1. [빠른 시작(Quick Start)](#빠른-시작quick-start)
2. [CLI 개요](#cli-개요)
3. [문법(DSL) 작성 규칙](#문법dsl-작성-규칙)

   * [지시어와 선언](#지시어와-선언)
   * [리터럴 키워드](#리터럴-키워드)
   * [규칙(EBNF)과 연산자](#규칙ebnf과-연산자)
   * [주석과 공백](#주석과-공백)
   * [세미콜론 규칙과 오류](#세미콜론-규칙과-오류)
4. [EBNF → BNF 전개 규칙](#ebnf--bnf-전개-규칙)
5. [파서 빌드 파이프라인](#파서-빌드-파이프라인)
6. [LALR vs SLR 차이와 예제](#lalr-vs-slr-차이와-예제)
7. [Rust 코드 방출과 사용법](#rust-코드-방출과-사용법)
8. [C 코드 방출과 사용법](#c-코드-방출과-사용법)
9. [런타임 렉서 인터페이스](#런타임-렉서-인터페이스)
10. [진단/디버깅/오류 메시지](#진단디버깅오류-메시지)
11. [자주 하는 질문(FAQ)](#자주-하는-질문faq)
12. [제한 사항과 향후 계획](#제한-사항과-향후-계획)

---

## 빠른 시작(Quick Start)

다음 예제는 **사칙연산** 문법을 LALR(1)으로 검사하고, Rust 파서 코드를 생성해 간단한 계산기(토이)를 실행하는 흐름을 보여줍니다.

### 1) 예제 문법 `expr.g`

```lepta
a# expr.g
%token NUMBER /-?(?:0|[1-9][0-9_]*)(?:\.[0-9_]+)?(?:[eE][+\-]?[0-9_]+)?/;
%ignore /\s+/;
"+" : "+"; "*" : "*"; "(" : "("; ")" : ")";

Expr   : Term ( "+" Term )*;
Term   : Factor ( "*" Factor )*;
Factor : NUMBER | "(" Expr ")";
```

### 2) 테이블 체크(기본 LALR)

```bash
python -m lepta.rascac check lepta/tests/grammar_test/expr.g -D
```

* `-D` : 디버그 출력(상태 수, 충돌, 상태 0의 아이템 등)
* `--parser slr` 로 SLR(1) 테이블도 비교 가능

### 3) Rust 코드 방출

```bash
python -m lepta.rascac build lepta/tests/grammar_test/expr.g \
  --lang rust -o lepta/tests/tmp/parser.rs -D
```

* `parser.rs` 파일 하나에 테이블과 파서 루프가 포함됩니다.
* 모듈명을 바꾸고 싶다면 `-m my_parser_mod` 옵션 사용.

### 4) 러너 예시 (`main.rs`)

```rust
mod parser; // 위에서 생성한 parser.rs와 동일 디렉터리
use parser::{parse, Lexer, EOF_TERM, TERM_NAMES};

struct SimpleLexer<'a> { s: &'a [u8], i: usize }
impl<'a> SimpleLexer<'a> {
    fn new(src: &'a str) -> Self { Self { s: src.as_bytes(), i: 0 } }
    fn skip_ws(&mut self) { while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() { self.i += 1; } }
    fn peek(&self) -> Option<char> { if self.i>=self.s.len() {None} else {Some(self.s[self.i] as char)} }
    fn bump(&mut self) -> Option<char> { let c = self.peek()?; self.i += 1; Some(c) }
    fn num(&mut self) -> bool { /* NUMBER 스캐너 구현(예제 생략) */ true }
}
impl<'a> Lexer for SimpleLexer<'a> {
    fn next_kind(&mut self) -> Option<u16> {
        self.skip_ws(); if self.i>=self.s.len(){ return Some(EOF_TERM) }
        match self.peek().unwrap() {
            '(' => { self.bump(); return Some(1) }
            ')' => { self.bump(); return Some(2) }
            '*' => { self.bump(); return Some(3) }
            '+' => { self.bump(); return Some(4) }
            _ => {}
        }
        if self.num() { return Some(5) } // NUMBER
        Some(EOF_TERM)
    }
}
fn main(){
    for s in ["1+2*3+(4*5)", "42", "(1+2)*3"] {
        let lx = SimpleLexer::new(s);
        println!("{:?} => {:?}", s, parse(lx));
    }
}
```

---

## CLI 개요

### 명령 형식

```
rascac check FILE.g [--parser {lalr|slr}] [-D]
rascac build FILE.g --lang {rust|c} -o OUT [--parser {lalr|slr}] [-m MOD] [-D]
```

### 옵션 설명

* `check` : 문법에서 **테이블만 생성**하여 상태 수/충돌을 확인합니다.
* `build` : 타깃 언어 코드로 **파서 소스**를 생성합니다.
* `--parser` : `lalr`(기본) 또는 `slr`를 선택.
* `--lang` : `rust` 또는 `c` 중 선택.
* `-o, --output` : 출력 파일 경로.
* `-m, --module` : (Rust) 모듈명 지정. 생략 시 출력 파일명에서 유도.
* `-D, --debug` : AST/BNF 요약, 상태 아이템 일부, 충돌 리포트를 상세 출력.

> **TIP**: `check`로 먼저 충돌 여부를 확인한 뒤 `build`를 수행하는 흐름을 추천합니다.

---

## 문법(DSL) 작성 규칙

### 지시어와 선언

* `%token NAME /regex/;` : **명명 토큰**을 선언합니다.
* `%ignore /regex/;` : 렉서가 **무시**할 패턴(공백/주석 등)을 선언합니다.
* `%start IDENT;` *(선택)* : 시작 비단말을 지정합니다. 생략 시 첫 규칙의 좌변이 시작심볼.

### 리터럴 키워드

* `"+" : "+";` 처럼 리터럴을 **단말 토큰**으로 고정합니다. 동일 스펠링은 하나의 토큰으로 합쳐집니다.

### 규칙(EBNF)과 연산자

* 생산 규칙: `A : α | β | ... ;`
* 토큰/비단말/그룹: `IDENT`, `"lit"`, `( subexpr )`
* **EBNF 수식어**: `?`(옵션), `*`(0+ 반복), `+`(1+ 반복)
* 우선순위는 **규칙 구조**로 표현(전형적 사다리: `Expr->Term`, `Term->Factor` 등)

### 주석과 공백

* 한 줄 주석: `// ...`
* 블록 주석: `/* ... */`
* 공백/주석은 `%ignore`로 토큰 스트림에서 제거됩니다.

### 세미콜론 규칙과 오류

* **모든 선언/규칙/키워드 매핑은 세미콜론(;)으로 끝나야 합니다.**
* 누락 시 에러 예:

  ```
  Missing ';' after %ignore declaration (semicolon is mandatory).
  - Found: STRING at 3:1
  - Example: %ignore /\s+/;
  ```
* 오류 위치는 문제 토큰 줄을 그대로 보여주고 `^` 캐럿으로 강조합니다.

---

## EBNF → BNF 전개 규칙

Rasca는 입력 문법을 내부적으로 **BNF**로 전개합니다.

* `X?` → `ε | X`
* `X*` → `ε | X X*` (좌재귀 전개 기본)
* `X+` → `X X*`
* `( ... )` → 보조 비단말 `__grpN` 생성
* 반복 보조 비단말: `__repN`

전개 후 **FIRST/FOLLOW**을 계산하고, 심볼 테이블을 고정합니다.

---

## 파서 빌드 파이프라인

1. **.g 로드** → **DSL 파싱(AST)**
2. **EBNF 전개(BNF)**
3. **심볼 테이블 고정(단말/비단말 ID 부여)**
4. **(선택) SLR**: LR(0) DFA + FOLLOW 기반 reduce
5. **(기본) LALR**: Canonical LR(1) → same-core 병합 → ACTION/GOTO
6. **테이블 압축/정규화**
7. **Rust/C 코드 방출** (테이블 + 런타임)

> LALR(1)은 LR(1) 상태에서 \*\*커널(core)\*\*이 같은 상태를 병합하면서, **lookahead 집합**을 커널 아이템 단위로 **합집합**하고 **closure**로 전파하여 SLR보다 정확한 reduce 결정을 수행합니다.

---

## LALR vs SLR 차이와 예제

다음 문법은 **LR(1) (→ LALR OK)** 이지만 **SLR(1)은 실패**하는 고전 사례입니다. Rasca에서 두 방식을 비교해 보며 차이를 확인하세요.

```lepta
%token ID /[A-Za-z_][A-Za-z0-9_]*/;
%ignore /\s+/;
"=" : "="; "*" : "*";

S : L "=" R | R ;
L : "*" R | ID ;
R : L ;
```

* **SLR 체크(충돌 기대)**

  ```bash
  python -m lepta.rascac check path/to/grammar.g --parser slr -D
  ```

  출력: `Conflicts: N (>0)` 및 `shift/reduce` 보고
* **LALR 체크(정상 기대)**

  ```bash
  python -m lepta.rascac check path/to/grammar.g --parser lalr -D
  ```

  출력: `Conflicts: 0`
  `-D` 상태 출력에서 `[A -> α · β , {a,b}]` 식의 **lookahead**를 확인할 수 있습니다.

---

## Rust 코드 방출과 사용법

### 방출

```bash
python -m lepta.rascac build FILE.g --lang rust -o out/parser.rs [-m module_name] [-D]
```

### 산출물 구조(요약)

* `const` 상수: 상태 수, 단말/비단말 수, 시작 상태, EOF 토큰 ID 등
* 테이블: `ACTION_ENTRIES`, `ACTION_OFFSETS`, `GOTO_ENTRIES`, `GOTO_OFFSETS`
* 규칙 테이블: `PROD_LHS[]`, `PROD_RHS_LEN[]`
* 심볼 문자열: `TERM_NAMES[]`, `NONTERM_NAMES[]`
* 런타임 Parse 루프: `parse<L: Lexer>(lx: L) -> Result<(), ParseError>`

### 런타임 `Lexer` 트레이트

```rust
pub trait Lexer { fn next_kind(&mut self) -> Option<u16>; }
```

* \*\*반드시 EOF 시 `Some(EOF_TERM)`\*\*를 반환(기본 런타임은 None도 EOF로 간주 가능하지만, 일관성을 위해 `EOF_TERM` 권장)
* 토큰 ID는 방출된 `TERM_NAMES` 순서와 일치해야 합니다.

### 에러 처리

* `ParseError { state, lookahead }`를 반환.
* 기대 집합은 “현재 상태 s에서 ACTION(s, a)가 존재하는 단말 a” 집합으로 복원 가능.

---

## C 코드 방출과 사용법

> 현재 IR과 런타임 템플릿을 C에도 연결할 수 있도록 설계되어 있습니다. (MVP에서는 Rust 경로 우선)

### 방출

```bash
python -m lepta.rascac build FILE.g --lang c -o out/parser.c -D
```

### 통합

* 생성된 `parser.c`는 정적 테이블과 파서 루프가 포함된 **단일 파일**입니다.
* 프로젝트에 포함하고, 사용자 구현 `lexer_next()`를 연결하여 동작시킵니다.

---

## 런타임 렉서 인터페이스

타깃별 렉서가 **다음 토큰의 kind(ID)** 를 돌려주는 간단한 pull 모델입니다.

### 공통 원칙

* `%ignore`로 선언한 패턴(공백, 주석 등)은 렉서에서 **소비**해야 합니다.
* **키워드 리터럴 우선**: IDENT를 읽은 뒤 **키워드 테이블**에서 리터럴과 일치하면 해당 단말 ID로 반환.
* 숫자/문자열/식별자 규격은 **문법의 %token** 선언 또는 기본 내장 토큰 규격을 따릅니다.

### 토큰 ID 매핑

* 방출물의 `TERM_NAMES` 순서를 출력하여(또는 문서를 확인하여) **ID ↔ 이름**을 구합니다.
* 예: `TERM_NAMES = ["$", "(", ")", "*", "+", "NUMBER"]` 이면, `"+"`의 ID는 4.

---

## 진단/디버깅/오류 메시지

### 디버그 출력(`-D`)

* AST 요약, BNF 요약(시작 기호/단말/비단말/규칙 수), 상태 수/충돌 수
* **상태 아이템** 일부(0번 상태):

  * SLR: `[A -> α · β]`
  * LALR: `[A -> α · β , {a,b}]` (lookahead 표시)

### 구문 오류 메시지

* 예시(런타임 파서):

  ```
  Parse error at 1:3: unexpected '*', expected one of {(, NUMBER}
  1+*2
    ^
  ```
* **세미콜론 누락**과 같은 문법 파일 오류도 **정확한 줄/열/예시**로 보고합니다.

### 충돌 보고

* `Conflicts: N`과 함께 각 항목을 `state S, on SYMBOL: shift / reduce` 형태로 리스팅.
* Rasca는 기본 정책으로 **shift 우선**, reduce/reduce는 **작은 규칙 인덱스 우선**을 채택합니다.

---

## 자주 하는 질문(FAQ)

**Q1. 왜 LALR이 기본인가요?**
A. SLR은 FOLLOW 기반이라 문맥을 구분하지 못해 불필요한 reduce가 발생할 수 있습니다. LALR은 lookahead 전파로 더 정확하며 보통의 실전 문법에 충분합니다.

**Q2. 액션 블록(세만틱 액션)은 어디에 쓰나요?**
A. v1에서는 기본 AST/노드 스텁을 생성하는 경량 모델입니다. v1.1에서 언어별 액션 블록 삽입을 지원할 예정입니다.

**Q3. 유니코드 식별자/XID는 지원하나요?**
A. 지원 계획에 포함되어 있으며, 타깃 코드로 **정적 범위 테이블**을 생성하는 방식을 권장합니다.

**Q4. 에러 복구는 가능한가요?**
A. MVP는 즉시 에러 보고(페일-패스트)입니다. 추후 panic-mode/동기화 토큰 기반 복구를 옵션으로 추가할 수 있습니다.

---

## 제한 사항과 향후 계획

* (MVP) C 경로는 IR 연결 작업이 남아 있습니다. Rust 경로는 완전 동작.
* (MVP) 액션 블록 미지원. v1.1에서 `@rs{..}@`, `@c{..}@` 형태 계획.
* 대규모 문법에서의 **테이블 압축 최적화**(공유 행/프리픽스) 추가 예정.
* **자기부트스트랩**(lepta.g를 Rasca로 파싱) 로드맵 포함.

---

## 부록 A. 문법 레퍼런스 요약

### 토큰/지시어

```
%token NAME /regex/;
%ignore /regex/;
%start START_SYMBOL;          # 선택
"literal" : "literal";      # 키워드 리터럴 고정
```

### 규칙

```
A : X Y? (Z+ | "lit")* ;
```

* `?` `*` `+` : 옵션/반복(EBNF)
* 그룹: `( ... )`
* 대안: `|`
* 모든 선언/규칙은 `;`로 종료

### 예약/네이밍

* 내부 전개에서 `__grpN`, `__repN` 같은 보조 비단말을 생성합니다. 사용자 심볼과 충돌하지 않도록 **밑줄 2개 접두어**는 사용하지 마세요.

---

## 부록 B. LALR vs SLR 검증 스크립트 힌트

```bash
# SLR
python -m lepta.rascac check tests/grammar_test/expr.g --parser slr -D

# LALR
python -m lepta.rascac check tests/grammar_test/expr.g --parser lalr -D

# Rust 방출(기본 LALR)
python -m lepta.rascac build tests/grammar_test/expr.g --lang rust -o tests/tmp/parser.rs -D
```

---

## 부록: 각 목차 보강 설명

아래는 본문 각 절에서 다룬 개념을 초보자 관점에서 다시 정리한 **보강 노트**입니다. 코드 예시는 별도 *lepta 예제 모음(실습 가이드)* 문서에 모아두었습니다.

### A. DSL 핵심 요약

* 문법 파일은 선언 영역과 규칙 영역으로 나뉩니다. 선언은 토큰/무시 패턴/시작기호를 정의하고, 규칙은 비단말 간의 생성 관계를 정의합니다.
* 리터럴 키워드는 토큰화 단계에서 고정된 심볼(예: '+', '\*')을 만듭니다. 같은 스펠링의 리터럴을 여러 번 선언해도 하나로 통합됩니다.
* EBNF 수식어(?, \*, +)는 작성은 간단하지만, 내부에서는 모두 BNF로 전개되어 동치의 순수 규칙 집합으로 바뀝니다.

### B. 파이프라인이 중요한 이유

* Rasca는 규칙을 곧바로 테이블로 바꾸지 않고, 중간에 **BNF 전개→FIRST/FOLLOW/NULLABLE 계산→상태기계 구성**을 거칩니다. 이 단계들이 잘 정의되어 있어야 충돌의 원인을 추적하고, 문법을 안정적으로 개선할 수 있습니다.
* LALR(1)은 canonical LR(1)의 정밀함과 상태 수의 실용적 균형을 맞춥니다. 실무 문법에서 충돌을 줄이는 데 큰 효과가 있습니다.

### C. 충돌을 읽는 법

* `shift / reduce`: 같은 상태·같은 단말에서 한쪽은 이동, 한쪽은 축약을 원함. 보통 우선순위·결합 규칙을 규칙 구조로 표현하거나, LALR을 사용하여 해소합니다.
* `reduce / reduce`: 서로 다른 규칙로 축약하려는 충돌. 문맥이 모호하거나 규칙이 과도하게 중첩된 경우가 많습니다.
* Rasca는 정책상 **shift 우선**으로 테이블을 구성하되, 충돌 보고를 상세히 남겨 원인 추적을 돕습니다.

### D. 런타임 렉서 작성 팁

* 무시 패턴은 렉서에서 실제로 소비해야 문법과 일관됩니다.
* 리터럴과 식별자가 겹칠 수 있을 때는 "리터럴 우선" 전략이 흔합니다(예: `if` 키워드는 IDENT가 아니라 키워드 토큰으로 인식).
* 토큰 ID 매핑은 방출물의 `TERM_NAMES`를 기준으로 고정됩니다. 개발 초기에 `TERM_NAMES`를 출력해 매핑을 확인하세요.

### E. 코드 방출 산출물 구조를 이해하면 디버깅이 쉬워집니다

* ACTION/GOTO 배열은 상태별 구간 오프셋을 통해 조회됩니다. 특정 상태에서 허용되는 단말이 무엇인지, 어떤 축약이 일어나는지 역으로 확인할 수 있습니다.
* 규칙 테이블(`PROD_LHS`, `PROD_RHS_LEN`)은 축약 시 스택에서 몇 개를 팝할지, 어떤 비단말로 이동할지를 알려줍니다.

### F. 한계와 현실적인 사용법

* 세만틱 액션이 없는 v1에서는 **구문 검증/파싱 성공 여부 판단/간단한 트리 스텁** 수준까지의 용도가 현실적입니다. 값 계산, 변수 해석, 템플릿 치환 등은 후속 버전의 액션 블록 또는 상위 호스트 코드가 담당합니다.
* 그럼에도 LALR 테이블과 런타임 루프가 일관되게 생성되므로, 문법 품질과 파서 안정성은 쉽게 확보할 수 있습니다.
