from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional


INT16_MOD = 1 << 16
INT16_MAX = (1 << 15) - 1
MEMORY_SIZE = 256


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Instruction:
    opcode: str
    args: tuple[str, ...]
    text: str


@dataclass
class MachineState:
    registers: List[int] = field(default_factory=lambda: [0] * 8)
    ip: int = 0
    z: int = 0
    n: int = 0
    c: int = 0
    memory: List[int] = field(default_factory=lambda: [0] * MEMORY_SIZE)
    output: Dict[int, int] = field(default_factory=dict)
    halted: int = 0
    error: str = "NONE"

    def clone(self) -> "MachineState":
        return MachineState(
            registers=self.registers.copy(),
            ip=self.ip,
            z=self.z,
            n=self.n,
            c=self.c,
            memory=self.memory.copy(),
            output=self.output.copy(),
            halted=self.halted,
            error=self.error,
        )

    def serialize(self) -> str:
        registers = " ".join(f"R{i}={value}" for i, value in enumerate(self.registers))
        out0 = self.output.get(0, "_")
        lines = [
            f"IP={self.ip}",
            registers,
            f"Z={self.z} N={self.n} C={self.c}",
            f"OUT[0]={out0}",
            f"HALTED={self.halted}",
            f"ERROR={self.error}",
        ]
        touched = [f"M[{idx}]={self.memory[idx]}" for idx in range(MEMORY_SIZE) if self.memory[idx] != 0]
        if touched:
            lines.append(f"MEM[{' '.join(touched)}]")
        return "\n".join(lines)


@dataclass
class Program:
    instructions: List[Instruction]
    labels: Dict[str, int]


@dataclass(frozen=True)
class TraceStep:
    step: int
    instruction: str
    state_text: str

    def serialize(self) -> str:
        return f"STEP {self.step}\nEXEC {self.instruction}\n{self.state_text}"


@dataclass(frozen=True)
class Transition:
    step: int
    instruction: str
    before_state_text: str
    after_state_text: str


REGISTER_RE = re.compile(r"^R([0-7])$")
LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DIRECT_ADDR_RE = re.compile(r"^\[(\-?\d+)\]$")
INPUT_RE = re.compile(r"^input\[(\d+)\]$")
OUTPUT_RE = re.compile(r"^output\[(\d+)\]$")

MVP_OPCODES = {
    "CONST": 2,
    "MOV": 2,
    "LOAD": 2,
    "STORE": 2,
    "READ": 2,
    "WRITE": 2,
    "ADD": 3,
    "SUB": 3,
    "CMP": 2,
    "TEST": 1,
    "JMP": 1,
    "JZ": 1,
    "JNZ": 1,
    "JL": 1,
    "JLE": 1,
    "JG": 1,
    "JGE": 1,
    "HALT": 0,
}


def wrap16(value: int) -> int:
    value &= INT16_MOD - 1
    if value > INT16_MAX:
        value -= INT16_MOD
    return value


def u16(value: int) -> int:
    return value & (INT16_MOD - 1)


def _split_operands(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(part.strip() for part in raw.split(","))


def _validate_register(token: str) -> None:
    if not REGISTER_RE.fullmatch(token):
        raise ParseError(f"invalid register: {token}")


def _validate_addr(token: str) -> None:
    match = DIRECT_ADDR_RE.fullmatch(token)
    if not match:
        raise ParseError(f"invalid memory operand: {token}")
    addr = int(match.group(1))
    if not 0 <= addr < MEMORY_SIZE:
        raise ParseError(f"memory address out of range: {addr}")


def _validate_input(token: str) -> None:
    if not INPUT_RE.fullmatch(token):
        raise ParseError(f"invalid input operand: {token}")


def _validate_output(token: str) -> None:
    if not OUTPUT_RE.fullmatch(token):
        raise ParseError(f"invalid output operand: {token}")


def _validate_int(token: str) -> None:
    try:
        int(token)
    except ValueError as exc:
        raise ParseError(f"invalid integer literal: {token}") from exc


def parse_program(source: str) -> Program:
    instructions: List[Instruction] = []
    labels: Dict[str, int] = {}

    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":"):
            label = line[:-1].strip()
            if not LABEL_RE.fullmatch(label):
                raise ParseError(f"line {lineno}: invalid label: {label}")
            if label in labels:
                raise ParseError(f"line {lineno}: duplicate label: {label}")
            labels[label] = len(instructions)
            continue

        parts = line.split(None, 1)
        opcode = parts[0].upper()
        if opcode not in MVP_OPCODES:
            raise ParseError(f"line {lineno}: unknown opcode: {opcode}")
        operands = _split_operands(parts[1] if len(parts) > 1 else "")
        if len(operands) != MVP_OPCODES[opcode]:
            raise ParseError(f"line {lineno}: wrong operand count for {opcode}")
        _validate_instruction(opcode, operands, lineno)
        instructions.append(Instruction(opcode=opcode, args=operands, text=_canonical_instruction(opcode, operands)))

    _validate_jump_labels(instructions, labels)
    return Program(instructions=instructions, labels=labels)


def _validate_instruction(opcode: str, operands: tuple[str, ...], lineno: int) -> None:
    if opcode == "CONST":
        _validate_register(operands[0])
        _validate_int(operands[1])
    elif opcode == "MOV":
        _validate_register(operands[0])
        _validate_register(operands[1])
    elif opcode == "LOAD":
        _validate_register(operands[0])
        _validate_addr(operands[1])
    elif opcode == "STORE":
        _validate_addr(operands[0])
        _validate_register(operands[1])
    elif opcode == "READ":
        _validate_register(operands[0])
        _validate_input(operands[1])
    elif opcode == "WRITE":
        _validate_output(operands[0])
        _validate_register(operands[1])
    elif opcode in {"ADD", "SUB"}:
        for operand in operands:
            _validate_register(operand)
    elif opcode == "CMP":
        _validate_register(operands[0])
        _validate_register(operands[1])
    elif opcode == "TEST":
        _validate_register(operands[0])
    elif opcode in {"JMP", "JZ", "JNZ", "JL", "JLE", "JG", "JGE"}:
        if not LABEL_RE.fullmatch(operands[0]):
            raise ParseError(f"line {lineno}: invalid label operand: {operands[0]}")
    elif opcode == "HALT":
        return
    else:
        raise ParseError(f"line {lineno}: unsupported opcode: {opcode}")


def _validate_jump_labels(instructions: List[Instruction], labels: Dict[str, int]) -> None:
    for inst in instructions:
        if inst.opcode in {"JMP", "JZ", "JNZ", "JL", "JLE", "JG", "JGE"} and inst.args[0] not in labels:
            raise ParseError(f"unresolved label: {inst.args[0]}")


def _canonical_instruction(opcode: str, operands: tuple[str, ...]) -> str:
    if operands:
        return f"{opcode} {', '.join(operands)}"
    return opcode


def initial_state(memory: Optional[Dict[int, int]] = None) -> MachineState:
    state = MachineState()
    if memory:
        for addr, value in memory.items():
            if not 0 <= addr < MEMORY_SIZE:
                raise ValueError(f"memory address out of range: {addr}")
            state.memory[addr] = wrap16(value)
    return state


def run_program(
    source: str,
    inputs: Optional[Dict[int, int]] = None,
    memory: Optional[Dict[int, int]] = None,
    max_steps: int = 1000,
    trace: bool = False,
) -> tuple[MachineState, List[TraceStep]]:
    program = parse_program(source)
    state = initial_state(memory)
    trace_steps: List[TraceStep] = []
    step_count = 0
    input_values = {k: wrap16(v) for k, v in (inputs or {}).items()}

    while not state.halted and state.error == "NONE":
        if step_count >= max_steps:
            state.error = "MAX_STEPS_EXCEEDED"
            break
        if state.ip < 0 or state.ip >= len(program.instructions):
            state.error = "IP_OOB"
            break

        instruction = program.instructions[state.ip]
        execute_instruction(state, instruction, program.labels, input_values)
        step_count += 1
        if trace:
            trace_steps.append(TraceStep(step=step_count, instruction=instruction.text, state_text=state.serialize()))

    return state, trace_steps


def collect_transitions(
    source: str,
    inputs: Optional[Dict[int, int]] = None,
    memory: Optional[Dict[int, int]] = None,
    max_steps: int = 1000,
) -> tuple[MachineState, List[Transition]]:
    program = parse_program(source)
    state = initial_state(memory)
    transitions: List[Transition] = []
    step_count = 0
    input_values = {k: wrap16(v) for k, v in (inputs or {}).items()}

    while not state.halted and state.error == "NONE":
        if step_count >= max_steps:
            state.error = "MAX_STEPS_EXCEEDED"
            break
        if state.ip < 0 or state.ip >= len(program.instructions):
            state.error = "IP_OOB"
            break

        instruction = program.instructions[state.ip]
        before = state.serialize()
        execute_instruction(state, instruction, program.labels, input_values)
        step_count += 1
        transitions.append(
            Transition(
                step=step_count,
                instruction=instruction.text,
                before_state_text=before,
                after_state_text=state.serialize(),
            )
        )

    return state, transitions


def execute_instruction(
    state: MachineState,
    instruction: Instruction,
    labels: Dict[str, int],
    input_values: Dict[int, int],
) -> None:
    op = instruction.opcode
    args = instruction.args

    if op == "CONST":
        state.registers[_reg_index(args[0])] = wrap16(int(args[1]))
        state.ip += 1
    elif op == "MOV":
        state.registers[_reg_index(args[0])] = state.registers[_reg_index(args[1])]
        state.ip += 1
    elif op == "LOAD":
        addr = _addr_value(args[1])
        if not 0 <= addr < MEMORY_SIZE:
            state.error = "OOB_MEMORY"
            return
        state.registers[_reg_index(args[0])] = state.memory[addr]
        state.ip += 1
    elif op == "STORE":
        addr = _addr_value(args[0])
        if not 0 <= addr < MEMORY_SIZE:
            state.error = "OOB_MEMORY"
            return
        state.memory[addr] = state.registers[_reg_index(args[1])]
        state.ip += 1
    elif op == "READ":
        index = _io_index(args[1], INPUT_RE)
        if index not in input_values:
            state.error = "OOB_INPUT"
            return
        state.registers[_reg_index(args[0])] = input_values[index]
        state.ip += 1
    elif op == "WRITE":
        index = _io_index(args[0], OUTPUT_RE)
        state.output[index] = state.registers[_reg_index(args[1])]
        state.ip += 1
    elif op == "ADD":
        a = state.registers[_reg_index(args[1])]
        b = state.registers[_reg_index(args[2])]
        raw = u16(a) + u16(b)
        result = wrap16(raw)
        state.registers[_reg_index(args[0])] = result
        _set_flags(state, result=result, carry=1 if raw > 0xFFFF else 0)
        state.ip += 1
    elif op == "SUB":
        a = state.registers[_reg_index(args[1])]
        b = state.registers[_reg_index(args[2])]
        raw = u16(a) - u16(b)
        result = wrap16(raw)
        state.registers[_reg_index(args[0])] = result
        _set_flags(state, result=result, carry=1 if u16(a) < u16(b) else 0)
        state.ip += 1
    elif op == "CMP":
        a = state.registers[_reg_index(args[0])]
        b = state.registers[_reg_index(args[1])]
        raw = u16(a) - u16(b)
        result = wrap16(raw)
        _set_flags(state, result=result, carry=1 if u16(a) < u16(b) else 0)
        state.ip += 1
    elif op == "TEST":
        value = state.registers[_reg_index(args[0])]
        _set_flags(state, result=value, carry=0)
        state.ip += 1
    elif op == "JMP":
        state.ip = labels[args[0]]
    elif op == "JZ":
        state.ip = labels[args[0]] if state.z == 1 else state.ip + 1
    elif op == "JNZ":
        state.ip = labels[args[0]] if state.z == 0 else state.ip + 1
    elif op == "JL":
        state.ip = labels[args[0]] if state.n == 1 else state.ip + 1
    elif op == "JLE":
        state.ip = labels[args[0]] if (state.n == 1 or state.z == 1) else state.ip + 1
    elif op == "JG":
        state.ip = labels[args[0]] if (state.n == 0 and state.z == 0) else state.ip + 1
    elif op == "JGE":
        state.ip = labels[args[0]] if (state.n == 0 or state.z == 1) else state.ip + 1
    elif op == "HALT":
        state.halted = 1
    else:
        state.error = "BAD_OPCODE"


def _set_flags(state: MachineState, result: int, carry: int) -> None:
    state.z = 1 if result == 0 else 0
    state.n = 1 if result < 0 else 0
    state.c = carry


def _reg_index(token: str) -> int:
    match = REGISTER_RE.fullmatch(token)
    if not match:
        raise ValueError(f"invalid register: {token}")
    return int(match.group(1))


def _addr_value(token: str) -> int:
    match = DIRECT_ADDR_RE.fullmatch(token)
    if not match:
        raise ValueError(f"invalid address: {token}")
    return int(match.group(1))


def _io_index(token: str, pattern: re.Pattern[str]) -> int:
    match = pattern.fullmatch(token)
    if not match:
        raise ValueError(f"invalid io operand: {token}")
    return int(match.group(1))
