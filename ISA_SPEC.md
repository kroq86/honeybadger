# AI-Assembly ISA v0 Specification

## 1. Purpose

This document defines the **formal MVP specification** for `AI-Assembly ISA v0`.

The goal of this ISA is not hardware compatibility or production efficiency. The goal is to provide a **small, deterministic, fully traceable execution model** suitable for:

- reference VM implementation
- synthetic dataset generation
- step-by-step supervision
- algorithmic reasoning experiments
- experiments toward execution-style behavior (results on this ISA may not generalize to other ISAs or task distributions)

This spec is intentionally strict. If an implementation differs from this document, the document wins.

## 2. Version Scope

This specification defines the **MVP execution profile** only.

Included instructions:

```text
CONST
MOV
LOAD
STORE
READ
WRITE
ADD
SUB
CMP
TEST
JMP
JZ
JNZ
JL
JLE
JG
JGE
HALT
```

Not included in MVP:

```text
MUL DIV MOD INC DEC
AND OR XOR NOT
PUSH POP CALL RET
NOP
```

These may be added in a later version, but are out of scope for `v0-mvp`.

## 3. Machine Model

### 3.1. Registers

The machine has exactly 8 general-purpose registers:

```text
R0 R1 R2 R3 R4 R5 R6 R7
```

All registers store signed 16-bit integers.

### 3.2. Word Size

The machine uses:

```text
signed int16
```

Value range:

```text
-32768 .. 32767
```

All arithmetic results are reduced modulo `2^16` and then interpreted as signed 16-bit integers.

Examples:

- `32767 + 1 -> -32768`
- `-32768 - 1 -> 32767`

### 3.3. Flags

The machine has three flags:

```text
Z N C
```

Definitions:

- `Z = 1` if the result of the relevant operation is exactly `0`, else `0`
- `N = 1` if the signed result of the relevant operation is negative, else `0`
- `C = 1` if the relevant operation produced unsigned carry or borrow according to the rules below, else `0`

### 3.4. Instruction Pointer

The machine has one instruction pointer:

```text
IP
```

`IP` is the zero-based index of the current instruction in the resolved program.

### 3.5. Memory

The machine has flat memory:

```text
M[0..255]
```

Memory contains 256 signed 16-bit cells.

### 3.6. Input and Output

Input and output are modeled as indexed arrays:

```text
IN[0..K]
OUT[0..K]
```

`IN` is provided at execution start.

`OUT` is initially empty. An unwritten output slot is considered unset.

### 3.7. Halt and Error

The machine state includes:

```text
HALTED: 0 | 1
ERROR: NONE | <ERROR_CODE>
```

If `ERROR != NONE`, the machine is in a terminal error state.

If `HALTED = 1`, the machine is in a terminal success state.

No instruction may execute once the machine is terminal.

## 4. Program Format

### 4.1. General Rules

- One instruction per line
- Labels are allowed
- Labels map to instruction indices after parsing
- Blank lines are allowed
- Comments are not part of the canonical training format

### 4.2. Canonical Syntax

Example:

```text
START:
READ R1, input[0]
READ R2, input[1]
ADD R3, R1, R2
WRITE output[0], R3
HALT
```

### 4.3. Label Syntax

Valid labels:

- must start with `A-Z` or `_`
- remaining chars may be `A-Z`, `a-z`, `0-9`, `_`

Examples:

- `LOOP`
- `END_1`
- `_START`

### 4.4. Register Syntax

Valid registers:

```text
R0 R1 R2 R3 R4 R5 R6 R7
```

Any other register token is invalid.

### 4.5. Address Syntax

For MVP, only direct integer addressing is allowed for `LOAD` and `STORE`:

```text
LOAD R1, [10]
STORE [20], R2
```

Dynamic addressing such as `[R1]` is not part of the MVP.

### 4.6. Input/Output Syntax

Canonical syntax:

```text
READ R1, input[0]
WRITE output[0], R2
```

Only non-negative integer indices are allowed.

## 5. Parsing and Resolution

### 5.1. Parse Phase

The parser must:

- tokenize each line
- validate opcode and operand count
- validate operand forms
- build a linear instruction list
- collect label definitions

### 5.2. Label Resolution

Labels resolve to instruction indices in the instruction list after label-only lines are removed.

Example:

```text
START:
CONST R1, 1
JMP START
```

Resolved instructions:

```text
0: CONST R1, 1
1: JMP 0
```

### 5.3. Parse Errors

The implementation must reject the program before execution if any of the following occur:

- unknown opcode
- wrong operand count
- invalid register
- invalid integer literal
- invalid label syntax
- duplicate label
- unresolved label
- invalid `input[k]` or `output[k]` syntax
- invalid `[addr]` syntax

## 6. State Initialization

At execution start:

- all registers are `0`
- all flags are `0`
- `IP = 0`
- all memory cells are `0` unless explicitly initialized by the runtime
- all output slots are unset
- `HALTED = 0`
- `ERROR = NONE`

Optionally, the runtime may provide initial memory contents. If so, that must be explicit in the task input and trace format.

## 7. Arithmetic Rules

### 7.1. Signed Conversion

All stored machine values must always be normalized to signed int16.

Define:

```text
wrap16(x) = ((x mod 65536) interpreted as signed int16)
```

### 7.2. Unsigned View

For carry detection, each operand may also be viewed as unsigned 16-bit:

```text
u16(x) in 0..65535
```

### 7.3. ADD Carry Rule

For `ADD dst, a, b`:

- compute `raw = u16(a) + u16(b)`
- `C = 1` if `raw > 65535`, else `0`
- stored result is `wrap16(raw)`

### 7.4. SUB Borrow Rule

For `SUB dst, a, b` and `CMP a, b`:

- compute `raw = u16(a) - u16(b)`
- `C = 1` if `u16(a) < u16(b)`, else `0`
- stored or compared result is `wrap16(raw)`

This treats `C` as unsigned borrow for subtraction-style operations.

## 8. Flag Update Policy

Only these instructions update flags in the current profile:

```text
ADD
SUB
CMP
TEST
```

Rules:

- `Z` and `N` are derived from the signed int16 result
- `C` follows the carry/borrow rules above

These instructions do **not** update flags:

```text
CONST MOV LOAD STORE READ WRITE JMP JZ JNZ JL JLE JG JGE HALT
```

## 9. Instruction Semantics

All semantics below assume execution begins in a non-terminal state.

### 9.1. `CONST dst, imm`

Operation:

- `dst := wrap16(imm)`

Effects:

- registers updated
- flags unchanged
- `IP := IP + 1`

Errors:

- none at runtime if parse succeeded

### 9.2. `MOV dst, src`

Operation:

- `dst := src`

Effects:

- destination register updated
- flags unchanged
- `IP := IP + 1`

### 9.3. `LOAD dst, [addr]`

Operation:

- `dst := M[addr]`

Effects:

- destination register updated
- flags unchanged
- `IP := IP + 1`

Errors:

- `OOB_MEMORY` if `addr < 0 or addr > 255`

Note:

In canonical MVP syntax, invalid addresses should normally be blocked at parse time if literal and out of range. Runtime handling still exists for robustness.

### 9.4. `STORE [addr], src`

Operation:

- `M[addr] := src`

Effects:

- memory updated
- flags unchanged
- `IP := IP + 1`

Errors:

- `OOB_MEMORY` if `addr < 0 or addr > 255`

### 9.5. `READ dst, input[k]`

Operation:

- `dst := IN[k]`

Effects:

- destination register updated
- flags unchanged
- `IP := IP + 1`

Errors:

- `OOB_INPUT` if `k` is not present in the provided input

### 9.6. `WRITE output[k], src`

Operation:

- `OUT[k] := src`

Effects:

- output updated
- flags unchanged
- `IP := IP + 1`

Errors:

- `OOB_OUTPUT` if the runtime enforces bounded output and `k` is outside the allowed range

Default MVP runtime policy:

- output is sparse and grows on write
- therefore `WRITE` does not error for non-negative `k`

### 9.7. `ADD dst, a, b`

Operation:

- `raw := u16(a) + u16(b)`
- `result := wrap16(raw)`
- `dst := result`

Flags:

- `Z := 1 if result == 0 else 0`
- `N := 1 if result < 0 else 0`
- `C := 1 if raw > 65535 else 0`

Effects:

- destination register updated
- flags updated
- `IP := IP + 1`

### 9.8. `SUB dst, a, b`

Operation:

- `raw := u16(a) - u16(b)`
- `result := wrap16(raw)`
- `dst := result`

Flags:

- `Z := 1 if result == 0 else 0`
- `N := 1 if result < 0 else 0`
- `C := 1 if u16(a) < u16(b) else 0`

Effects:

- destination register updated
- flags updated
- `IP := IP + 1`

### 9.9. `CMP a, b`

Operation:

- compute subtraction result exactly as in `SUB`, but do not store it

Flags:

- `Z := 1 if result == 0 else 0`
- `N := 1 if result < 0 else 0`
- `C := 1 if u16(a) < u16(b) else 0`

Effects:

- registers unchanged
- memory unchanged
- `IP := IP + 1`

### 9.10. `TEST a`

Operation:

- compute `result := a`

Flags:

- `Z := 1 if result == 0 else 0`
- `N := 1 if result < 0 else 0`
- `C := 0`

Effects:

- registers unchanged
- memory unchanged
- `IP := IP + 1`

### 9.11. `JMP label`

Operation:

- `IP := resolved_label_index`

Effects:

- only `IP` changes
- flags unchanged

### 9.12. `JZ label`

Operation:

- if `Z == 1`, then `IP := resolved_label_index`
- else `IP := IP + 1`

Effects:

- flags unchanged

### 9.13. `JNZ label`

Operation:

- if `Z == 0`, then `IP := resolved_label_index`
- else `IP := IP + 1`

Effects:

- flags unchanged

### 9.14. `JL label`

Operation:

- if `N == 1`, then `IP := resolved_label_index`
- else `IP := IP + 1`

Effects:

- flags unchanged

### 9.15. `JLE label`

Operation:

- if `N == 1 or Z == 1`, then `IP := resolved_label_index`
- else `IP := IP + 1`

Effects:

- flags unchanged

### 9.16. `JG label`

Operation:

- if `N == 0 and Z == 0`, then `IP := resolved_label_index`
- else `IP := IP + 1`

Effects:

- flags unchanged

### 9.17. `JGE label`

Operation:

- if `N == 0 or Z == 1`, then `IP := resolved_label_index`
- else `IP := IP + 1`

Effects:

- flags unchanged

### 9.18. `HALT`

Operation:

- `HALTED := 1`

Effects:

- machine becomes terminal
- registers, memory, flags, output unchanged
- `IP` remains at the `HALT` instruction index in the terminal state

## 10. Execution Loop

Execution proceeds as follows:

1. If `HALTED = 1` or `ERROR != NONE`, stop.
2. If `IP < 0` or `IP >= program_length`, set `ERROR := IP_OOB` and stop.
3. Fetch instruction at `IP`.
4. Execute according to this spec.
5. If step counter exceeds configured maximum, set `ERROR := MAX_STEPS_EXCEEDED` and stop.
6. Repeat.

## 11. Error Codes

The MVP runtime must support these terminal error codes:

```text
NONE
OOB_MEMORY
OOB_INPUT
OOB_OUTPUT
BAD_OPCODE
BAD_OPERAND
IP_OOB
MAX_STEPS_EXCEEDED
```

For the strict reference implementation, parse-time failures should normally prevent execution and be reported separately. `BAD_OPCODE` and `BAD_OPERAND` exist mainly for defensive runtime behavior and corrupt input handling.

## 12. Canonical State Format

The canonical textual state format is:

```text
IP=<int>
R0=<int> R1=<int> R2=<int> R3=<int> R4=<int> R5=<int> R6=<int> R7=<int>
Z=<0|1> N=<0|1> C=<0|1>
OUT[0]=<value_or_underscore>
HALTED=<0|1>
ERROR=<ERROR_CODE>
```

If memory must be shown, canonical compact format is:

```text
MEM[M[3]=7 M[10]=-1]
```

Rules:

- touched memory cells only
- increasing address order
- omit `MEM[...]` if no cells were written or explicitly initialized

## 13. Canonical Trace Format

One execution step is serialized as:

```text
STEP <n>
EXEC <instruction_text>
IP=<int>
R0=<int> R1=<int> R2=<int> R3=<int> R4=<int> R5=<int> R6=<int> R7=<int>
Z=<0|1> N=<0|1> C=<0|1>
OUT[0]=<value_or_underscore>
HALTED=<0|1>
ERROR=<ERROR_CODE>
```

If memory is present, append:

```text
MEM[...]
```

The `EXEC` line must use canonical instruction formatting.

## 14. Canonical Instruction Formatting

The serializer must emit instructions exactly in one of these forms:

```text
CONST Rn, imm
MOV Rn, Rm
LOAD Rn, [addr]
STORE [addr], Rn
READ Rn, input[k]
WRITE output[k], Rn
ADD Rn, Ra, Rb
SUB Rn, Ra, Rb
CMP Ra, Rb
TEST Ra
JMP LABEL
JZ LABEL
JNZ LABEL
JL LABEL
JLE LABEL
JG LABEL
JGE LABEL
HALT
```

## 14.1 Ordered-Conditional Semantics Note

The ordered jumps `JL/JLE/JG/JGE` use the current `Z` and `N` flags only.

This means:

- they operate on the most recent wrapped signed comparison state
- they are deterministic and simple for learning
- they are **not** x86-style overflow-aware signed branches

Practical rule:

- use `CMP` or `TEST` immediately before ordered jumps
- treat them as ISA-native ordered predicates over the current int16 state, not as hardware-compatible condition codes

Whitespace rules:

- one space after opcode
- comma followed by one space
- no extra spaces

## 15. Worked Example

Program:

```text
READ R1, input[0]
READ R2, input[1]
ADD R3, R1, R2
WRITE output[0], R3
HALT
```

Input:

```text
input[0]=7
input[1]=5
```

Initial state:

```text
IP=0
R0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0
Z=0 N=0 C=0
OUT[0]=_
HALTED=0
ERROR=NONE
```

After step 1:

```text
STEP 1
EXEC READ R1, input[0]
IP=1
R0=0 R1=7 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0
Z=0 N=0 C=0
OUT[0]=_
HALTED=0
ERROR=NONE
```

After step 2:

```text
STEP 2
EXEC READ R2, input[1]
IP=2
R0=0 R1=7 R2=5 R3=0 R4=0 R5=0 R6=0 R7=0
Z=0 N=0 C=0
OUT[0]=_
HALTED=0
ERROR=NONE
```

After step 3:

```text
STEP 3
EXEC ADD R3, R1, R2
IP=3
R0=0 R1=7 R2=5 R3=12 R4=0 R5=0 R6=0 R7=0
Z=0 N=0 C=0
OUT[0]=_
HALTED=0
ERROR=NONE
```

After step 4:

```text
STEP 4
EXEC WRITE output[0], R3
IP=4
R0=0 R1=7 R2=5 R3=12 R4=0 R5=0 R6=0 R7=0
Z=0 N=0 C=0
OUT[0]=12
HALTED=0
ERROR=NONE
```

After step 5:

```text
STEP 5
EXEC HALT
IP=4
R0=0 R1=7 R2=5 R3=12 R4=0 R5=0 R6=0 R7=0
Z=0 N=0 C=0
OUT[0]=12
HALTED=1
ERROR=NONE
```

## 16. Reference VM Requirements

A conforming reference VM must:

- reject invalid programs deterministically
- execute valid programs exactly according to this spec
- produce canonical terminal states
- produce canonical traces when trace mode is enabled
- enforce `max_steps`
- preserve determinism across runs

## 17. Deferred Decisions for v1+

The following design areas are intentionally deferred:

- dynamic memory addressing
- stack semantics
- subroutine semantics
- multiplication and division edge cases
- bitwise flag policy
- structured control-flow sugar
- compressed trace formats
- alternative IR forms

## 18. Immediate Next Step

With this specification frozen, the next artifact should be:

```text
python-vm-reference-plan.md
```

or directly:

```text
reference_vm.py + tests
```
