# lepta 예제 01: REPL 사칙연산 계산기 (Rust)

> 이 문서는 **lepta**로 문법을 정의하고, Rust 파서 코드(parser.rs)를 생성한 뒤,
> 간단한 **REPL 계산기**(main.rs)를 붙여 완성하는 과정을 단계별로 안내합니다.
>
> 포인트
>
> * 우선순위/결합 규칙이 적용된 **수식 문법** (지수 `^`, 곱셈/나눗셈, 덧셈/뺄셈, 괄호, 단항 `-`).
> * **연속 연산자 금지** 같은 문법 외 규칙은 **렉서에서 즉시 에러**로 처리.
> * **세만틱 액션**으로 f64 값을 계산해 출력.
>
> 준비물: Python 3.10+, Rust (rustc)

---

## 1) 폴더 구조(권장)

```
/your-project
├─ lepta/                 # lepta 패키지 (이미 보유)
├─ expr.g                 # 이번 예제의 문법 파일
├─ parser.rs              # lepta가 생성할 Rust 파서 (빌드 단계에서 생성)
└─ main.rs                # REPL 계산기 (우리가 작성)
```

> `lepta` 패키지 루트에서 커맨드를 실행한다고 가정합니다. 다른 경로에서 실행할 경우 `expr.g` 경로만 맞춰주세요.

---

## 2) 문법 파일: `expr.g`

아래 내용을 **그대로** `expr.g` 로 저장하세요.

```ebnf
%token NUMBER /(?>0|[1-9][0-9_]*)(?:\.[0-9_]+)?(?:[eE][+\-]?[0-9_]+)?/;
%ignore /\s+/;
"(":"("; ")":")"; "+":"+"; "-":"-"; "*":"*"; "/":"/"; "^":"^";

%use precedence cpp_core;   // c++의 우선순위 템플릿 사용
%right UMINUS;              // 단항 - 은 가장 강한 레벨

Expr
  : Expr "+" Term
  | Expr "-" Term
  | Term
  ;

Term
  : Term "*" Pow
  | Term "/" Pow
  | Pow
  ;

Pow
  : Factor "^" Pow
  | Factor
  ;

Primary
  : NUMBER
  | "(" Expr ")"
  ;

Factor
  : "-" Factor %prec UMINUS
  | Primary
  ;
```

### 설명

* **우선순위 템플릿** `cpp_core` + 추가 `%right UMINUS` 로, `UMINUS`(단항 -)가 **가장 강한 결합**이 되게 합니다.
* `Pow`는 오른쪽 결합(`a ^ b ^ c` → `a ^ (b ^ c)`)이 되도록 규칙을 배치했습니다.
* `Primary`/`Factor` 분리로 단항 `-`와 괄호 우선순위를 직관적으로 표현합니다.

> **연속 연산자 금지(예: `1 ++ 8`)** 는 문법으로만 막기 어렵습니다. 이 예제에서는 **렉서에서 즉시 에러**로 처리합니다(아래 `main.rs`).

---

## 3) 문법 점검 & 파서 생성

### (1) 점검

```bash
python -m lepta.leptac check ./expr.g --parser lalr -D
```

* AST/BNF/파서 상태/충돌 여부 등을 출력합니다.
* `Conflicts: 0` 이면 정상.

### (2) Rust 파서 생성

```bash
python -m lepta.leptac build ./expr.g --parser lalr --lang rust -o ./parser.rs -D
```

* 현재 디렉터리에 `parser.rs`가 생성됩니다.
* 이 파일에는:

  * **Dense 테이블 기반 LR 런타임**(O(1) 조회)
  * **`parse_with_actions`**(세만틱 액션용)
  * **디버그용 심볼 이름 테이블**
  * (옵션) 아주 단순한 내장 샘플 렉서가 포함될 수 있지만, 이번 예제에선 **직접 만든 렉서**를 사용합니다.

---

## 4) REPL 계산기: `main.rs`

아래 코드를 **그대로** `main.rs` 로 저장하세요.

* 역할: 간단한 **렉서(ReplLexer)** + **액션(EvalActions)** + **REPL 루프**
* 특징:

  * **연속 연산자 금지**: `+ - * / ^`가 연속해서 나오면 렉서에서 즉시 에러
  * 단항 `-`는 **입력 시작** 또는 **`(` 뒤**에서만 허용

```rust
mod parser; // 같은 디렉터리에 생성된 parser.rs가 있어야 합니다.

use parser::{parse_with_actions, Lexer2, Token, TERM_NAMES, ACTION_KIND, N_TERMS};

/// term 이름 → term ID(u16)
fn tid(name: &str) -> u16 {
    for (i, n) in TERM_NAMES.iter().enumerate() {
        if *n == name { return i as u16; }
    }
    panic!("unknown terminal name: {name}");
}

fn is_digit(c: char) -> bool { c.is_ascii_digit() }
fn is_ws(c: char) -> bool { c.is_whitespace() }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Last {
    Start,      // 입력 시작/리셋 직후
    AfterLPar,  // '(' 직후
    AfterOp,    // 연산자 직후(+, -, *, /, ^)
    AfterValue, // 값/닫는 괄호 직후
}

/// 아주 단순한 표현식용 렉서 + “연속 연산자 금지”
/// - 공백 스킵
/// - '(', ')', '+', '-', '*', '/', '^' 인식
/// - NUMBER: 10진수 [0-9][0-9_]* (선택) + 소수부/지수부(선택)
/// - 단항 - 허용 위치: 입력 시작 / '(' 직후만 허용(그 외 연속 연산자는 에러)
struct ReplLexer<'a> {
    s: &'a str,
    i: usize,
    last: Last,
}

impl<'a> ReplLexer<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, i: 0, last: Last::Start }
    }

    fn peek(&self) -> Option<char> { self.s[self.i..].chars().next() }
    fn bump(&mut self) -> Option<char> {
        if self.i >= self.s.len() { return None; }
        let ch = self.peek()?; self.i += ch.len_utf8(); Some(ch)
    }
    fn skip_ws(&mut self) { while let Some(c) = self.peek() { if is_ws(c) { self.bump(); } else { break; } } }
    fn take_while<F: Fn(char)->bool>(&mut self, f: F, buf: &mut String) {
        while let Some(c) = self.peek() { if f(c) { buf.push(c); self.bump(); } else { break; } }
    }
    fn starts_with(&self, s: &str) -> bool { self.s[self.i..].as_bytes().starts_with(s.as_bytes()) }

    fn number(&mut self) -> String {
        // 정수부
        let mut buf = String::new();
        self.take_while(|c| c.is_ascii_digit() || c == '_', &mut buf);
        // .소수부
        if self.starts_with(".") {
            let save = self.i; let mut tmp = String::from("."); self.bump();
            let mut had = false; while let Some(c) = self.peek() { if c.is_ascii_digit() || c == '_' { had = true; tmp.push(c); self.bump(); } else { break; } }
            if had { buf.push_str(&tmp); } else { self.i = save; }
        }
        // 지수부
        if matches!(self.peek(), Some('e'|'E')) {
            let save = self.i; let mut tmp = String::new(); tmp.push(self.bump().unwrap());
            if matches!(self.peek(), Some('+'|'-')) { tmp.push(self.bump().unwrap()); }
            let mut had = false; while let Some(c) = self.peek() { if c.is_ascii_digit() || c == '_' { had = true; tmp.push(c); self.bump(); } else { break; } }
            if had { buf.push_str(&tmp); } else { self.i = save; }
        }
        buf
    }

    fn is_operator(ch: char) -> bool { matches!(ch, '+'|'-'|'*'|'/'|'^') }
}

impl<'a> Lexer2 for ReplLexer<'a> {
    fn next(&mut self) -> Option<Token> {
        self.skip_ws();
        let ch = self.peek()?;

        // 괄호는 항상 허용
        if ch == '(' { self.bump(); self.last = Last::AfterLPar; return Some(Token{kind: tid("("), text: "(".into()}); }
        if ch == ')' { self.bump(); self.last = Last::AfterValue; return Some(Token{kind: tid(")"), text: ")".into()}); }

        // 연산자
        if Self::is_operator(ch) {
            match self.last {
                Last::AfterOp => panic!("lex error: two operators in a row near byte {}", self.i),
                _ => {}
            }
            let s = ch.to_string();
            self.bump();
            self.last = Last::AfterOp;
            return Some(Token { kind: tid(&s), text: s });
        }

        // NUMBER
        if ch.is_ascii_digit() {
            let txt = self.number();
            self.last = Last::AfterValue;
            return Some(Token { kind: tid("NUMBER"), text: txt });
        }

        // 그 외: 에러
        panic!("lex error: unexpected char {:?} at byte {}", ch, self.i);
    }
}

/// 계산기 액션: f64로 평가
struct EvalActions;

impl parser::Actions for EvalActions {
    type Val = f64;

    fn on_shift(&mut self, tok: &Token) -> Self::Val {
        if TERM_NAMES[tok.kind as usize] == "NUMBER" {
            let clean = tok.text.replace('_', "");
            clean.parse::<f64>().expect("invalid NUMBER literal")
        } else { 0.0 }
    }

    fn on_reduce(&mut self, p: usize, _lhs: u16, rhs: &[Self::Val]) -> Self::Val {
        match p {
            0 => unreachable!(),               // __S' -> Expr
            // Expr
            1 => rhs[0] + rhs[2],              // Expr + Term
            2 => rhs[0] - rhs[2],              // Expr - Term
            3 => rhs[0],                       // Expr -> Term
            // Term
            4 => rhs[0] * rhs[2],              // Term * Pow
            5 => rhs[0] / rhs[2],              // Term / Pow
            6 => rhs[0],                       // Term -> Pow
            // Pow (오른쪽 결합)
            7 => rhs[0].powf(rhs[2]),          // Factor ^ Pow
            8 => rhs[0],                       // Pow -> Factor
            // Primary
            9  => rhs[0],                      // NUMBER
            10 => rhs[1],                      // '(' Expr ')'
            // Factor
            11 => -rhs[1],                     // '-' Factor (단항 -)
            12 => rhs[0],                      // Primary
            _ => unreachable!("unknown production {p}"),
        }
    }
}

/// 결과 포매팅: 정수처럼 떨어지면 소수점 제거
fn fmt_num(x: f64) -> String {
    if x.is_finite() && (x.fract().abs() < 1e-12) {
        format!("{}", x.round() as i128)
    } else {
        let s = format!("{:.12}", x);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn main() {
    println!("lepta REPL calculator");
    println!(" - operators: + - * / ^   (unary - only at start or after '(')");
    println!(" - examples:  1+2*3^(4-5),  (-7),  2^-3");
    println!(" - type 'quit' or 'exit' or press Enter on empty line to leave.\n");

    let mut line = String::new();
    let stdin = std::io::stdin();

    loop {
        line.clear();
        print!("> ");
        use std::io::Write; std::io::stdout().flush().ok();

        if stdin.read_line(&mut line).is_err() { break; }
        let src = line.trim();
        if src.is_empty() || src.eq_ignore_ascii_case("quit") || src.eq_ignore_ascii_case("exit") { break; }

        let mut lx = ReplLexer::new(src);
        let mut acts = EvalActions;

        // 렉서/파서 오류를 구분하여 보기 좋게 출력
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            parse_with_actions(lx, &mut acts)
        }));

        match result {
            Err(payload) => {
                // 렉서 단계 패닉(연속 연산자 등)
                if let Some(msg) = payload.downcast_ref::<&str>() {
                    eprintln!("Lex error: {msg}");
                } else if let Some(msg) = payload.downcast_ref::<String>() {
                    eprintln!("Lex error: {msg}");
                } else {
                    eprintln!("Lex error: unknown");
                }
            }
            Ok(Err(e)) => {
                // 파서 에러: 예상 집합도 출력
                let mut expected: Vec<&'static str> = Vec::new();
                for t in 0..N_TERMS { let idx = e.state * N_TERMS + t; if ACTION_KIND[idx] != 0 { expected.push(TERM_NAMES[t]); } }
                eprintln!(
                    "Parse error at state={}, lookahead={} ('{}')\n  expected: {:?}",
                    e.state, e.lookahead, TERM_NAMES[e.lookahead as usize], expected
                );
            }
            Ok(Ok(val)) => println!("{}", fmt_num(val)),
        }
    }
}
```

---

## 5) 빌드 & 실행

### (1) 컴파일

```bash
rustc main.rs -o repl_calc
```

> 같은 디렉터리에 `parser.rs` 가 있어야 합니다(3단계에서 생성됨).

### (2) 실행

```bash
./repl_calc
```

샘플 세션 예시:

```
lepta REPL calculator
 - operators: + - * / ^   (unary - only at start or after '(')
 - examples:  1+2*3^(4-5),  (-7),  2^-3
 - type 'quit' or 'exit' or press Enter on empty line to leave.

> 1+2*3^(4-5)
-5
> 1 - -7
Lex error: two operators in a row near byte 4
> (1 + 2) * 3
9
> quit
```

> 출력은 f64로 계산되며, 정수처럼 떨어지는 경우 소수점은 생략하여 보여줍니다.

---

## 6) 동작 원리 요약

* `parser.rs`에는 **LR 파서 런타임 + 테이블**과 함께, 세만틱 액션을 연결하는 `parse_with_actions`가 들어 있습니다.
* `main.rs`의 `EvalActions`는 `Actions` 트레이트를 구현하여, 각 프로덕션 축약 시 실제 값을 계산합니다.
* **연속 연산자 금지**는 문법 대신 **렉서의 상태(`last`)** 로 판정하여, `+ - * / ^`가 연속으로 등장하면 즉시 에러를 발생시킵니다.
* 단항 `-`는 입력 시작/괄호 뒤에서만 허용하므로, `1 - - 8`, `1 ++ 8`, `1*-2` 등이 렉서 단계에서 막힙니다.

---

## 7) 확장 아이디어

* **변수/대입/함수 호출**을 추가하고, 심볼 테이블과 에러 메시지를 확장.
* **우선순위 템플릿**을 커스터마이징(예: `precedence_templates.py` 수정)하여 연산자 추가.
* `main.rs`의 렉서를 개선해 **위치(행/열) 정보**, **오류 하이라이트** 등을 제공.
* 세만틱 액션에서 **AST 구축** → 이후 별도의 평가기/최적화기를 붙이는 구조로 확장.

---

## 8) 트러블슈팅

* **trait bound 오류(`Lexer2` 미구현)**: `parse_with_actions` 호출 시 `&mut lx`가 아니라 **값**을 넘기세요. `match parse_with_actions(lx, &mut acts)` 처럼.
* **프로덕션 인덱스가 다른가요?**: `-D`로 빌드/체크하면 프로덕션 순서를 확인할 수 있습니다. 본 예제의 `on_reduce`는 생성된 `parser.rs`의 인덱스(0\~12)를 기준으로 작성되었습니다.
* **`1 ++ 8`이 파서 에러로 나오나요?**: 렉서가 먼저 잡도록 `ReplLexer`를 그대로 사용했는지 확인하세요.

---

## 9) 요약

1. `expr.g` 저장 → `check`로 검증 → `build`로 `parser.rs` 생성.
2. `main.rs` 저장(위 코드 그대로).
3. `rustc main.rs -o repl_calc`로 컴파일.
4. `./repl_calc` 실행 → REPL에서 수식 입력해 계산.

