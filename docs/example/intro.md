# lepta 예제 모음 (실습 가이드)

본 문서는 **usage.md**를 보완하는 실습형 자료입니다. 코드 방출과 런타임 연결은 필요 최소한만 안내하고, 핵심은 **`.g` 문법만으로 문법 작성·테이블 생성·파서 코드 방출**을 반복해 보는 데 있습니다.

* 실습 1: 사칙연산 계산기(구문)
* 실습 2: 간단한 변수 대입/참조 문법(구문)
* 보너스: SLR과 LALR 차이를 드러내는 문법

> 주의: lepta v1은 세만틱 액션이 없으므로, 여기서는 **구문 분석 성공 여부와 트리 골격**에 집중합니다. 실제 값 계산이나 템플릿 치환은 후속 버전(액션 블록) 또는 상위 애플리케이션 코드에서 수행합니다.

---

## 실습 1. 사칙연산 계산기(구문)

### 1) 문법 파일 `expr.g`

```lepta
%token NUMBER /-?(?:0|[1-9][0-9_]*)(?:\.[0-9_]+)?(?:[eE][+\-]?[0-9_]+)?/;
%ignore /\s+/;
"+" : "+"; "-" : "-"; "*" : "*"; "/" : "/"; "(" : "("; ")" : ")";

Expr   : Term ( ("+"|"-") Term )*;
Term   : Factor ( ("*"|"/") Factor )*;
Factor : NUMBER | "(" Expr ")";
```

### 2) 테이블 검사 & 방출

```bash
python -m lepta.rascac check expr.g -D
python -m lepta.rascac build expr.g --lang rust -o parser.rs -D
```

### 3) 기대 사항

* LALR 기본 모드에서 충돌이 없어야 합니다.
* 방출된 `TERM_NAMES`에는 `$ ( ) * + - / NUMBER` 순(또는 유사)이 포함됩니다.
* 러너(간단 렉서)를 붙이면 `1+2*3` 같은 입력이 구문상 정상으로 인식됩니다.

> 값 계산(산술 평가)은 v1에서 문법만으로는 수행하지 않습니다. v1.1의 액션 블록 또는 상위 코드에서 AST 방문자로 구현하세요.

---

## 실습 2. 변수 대입/참조 문법(구문)

> 목표: **.g 파일만**으로 변수의 대입 구문과 참조 구문을 파싱합니다. (실제 치환/평가는 하지 않음)

### 1) 변수 대입 `assign.g`

```lepta
%token ID /[A-Za-z_][A-Za-z0-9_]*/;
%token NUMBER /[0-9]+/;
%ignore /\s+/;
"=" : "="; ";" : ";"; "+" : "+"; "-" : "-"; "*" : "*"; "/" : "/";

Prog   : Stmt* ;
Stmt   : ID "=" Expr ";" ;
Expr   : Term ( ("+"|"-") Term )* ;
Term   : Factor ( ("*"|"/") Factor )* ;
Factor : NUMBER | ID ;
```

* 이 문법은 `x=3; y=2; z=x+10;` 같은 **대입문 리스트**를 파싱합니다.
* 실제 이름 해석/값 계산은 후속 단계(액션 블록/호스트 코드)에서 수행합니다.

### 2) 변수 참조 포함 텍스트(템플릿 토큰화) `template_tokens.g`

```lepta
%token TEXT /[^{}\n]+/;   // 중괄호와 줄바꿈을 제외한 평문
%token ID   /[A-Za-z_][A-Za-z0-9_]*/;
%ignore /[ \t\r\n]+/;    // 공백/개행 무시(원한다면 제거)
"{{" : "{{"; "}}" : "}}";

Doc   : Chunk* ;
Chunk : TEXT | Ref ;
Ref   : "{{" ID "}}" ;
```

* 이 문법은 `Hello, {{name}}!` 같은 문자열에서 **변수 참조 자리**를 구문으로 식별합니다.
* 치환(실제 렌더링)은 후속 단계에서 변수 환경을 연결해 처리합니다.

### 3) 빌드

```bash
python -m lepta.rascac check assign.g -D
python -m lepta.rascac build assign.g --lang rust -o assign_parser.rs -D

python -m lepta.rascac check template_tokens.g -D
python -m lepta.rascac build template_tokens.g --lang rust -o template_parser.rs -D
```

### 4) 확장 아이디어

* `Stmt`에 `let` 키워드, 세미콜론 생략 규칙, 블록 스코프 등을 추가해 보세요.
* 템플릿 쪽은 `{{#if}}/{{/if}}`, `{{#each}}` 같은 블록 구조를 추가해 보세요(EBNF 수식어로 쉽게 스케치 가능).

---

## 보너스. SLR과 LALR 차이를 드러내는 문법

### 문법 `lalr_vs_slr.g`

```lepta
%token ID /[A-Za-z_][A-Za-z0-9_]*/;
%ignore /\s+/;
"=" : "="; "*" : "*";

S : L "=" R | R ;
L : "*" R | ID ;
R : L ;
```

### 실험

```bash
python -m lepta.rascac check lalr_vs_slr.g --parser slr -D   # 보통 충돌 발생
python -m lepta.rascac check lalr_vs_slr.g --parser lalr -D  # 충돌 0
```

### 관찰 포인트

* SLR은 FOLLOW 기반으로 reduce를 허용해 **불필요한 reduce**가 발생합니다.
* LALR은 canonical LR(1) lookahead를 커널 단위로 병합해 **정확한 reduce**만 허용합니다.

---

## 마무리

* `.g`만으로도 문법을 빠르게 스케치하고, LALR 테이블을 확인하는 데 Rasca는 매우 가볍습니다.
* 값 계산/치환은 후속 릴리스(액션 블록) 또는 상위 코드에서 연결하세요. 그 전에도 문법의 안정성, 충돌 여부, 토큰/규칙의 품질을 충분히 검증할 수 있습니다.
