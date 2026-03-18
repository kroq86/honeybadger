from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib import error, request


DEFAULT_STAGES = ["single_step", "next_2_steps"]
DEFAULT_HTTP_TIMEOUT_SECONDS = 12
DEFAULT_SHORT_TRACE_STEP_CAP = 6


@dataclass(frozen=True)
class Example:
    dataset_type: str
    program_name: str
    category: str
    prompt: str
    target: str
    split_family: str | None = None
    input_values: Dict[str, int] | None = None
    instruction: str | None = None
    num_steps: int | None = None
    selection_key: str | None = None


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        temperature: float = 0.0,
        num_predict: int = 1024,
        timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, system: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"failed to call ollama generate api: {exc}") from exc
        return str(parsed.get("response", "")).strip()


def check_ollama_available(host: str, timeout_seconds: int) -> tuple[bool, str]:
    req = request.Request(
        url=f"{host.rstrip('/')}/api/tags",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        return False, f"ollama_unavailable: {exc}"
    model_names = [model.get("name", "") for model in parsed.get("models", [])]
    return True, ", ".join(name for name in model_names if name)


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split(dataset_root: Path, stage: str, split: str) -> List[Example]:
    records = read_jsonl(dataset_root / stage / f"{split}.jsonl")
    return [
        Example(
            dataset_type=record["dataset_type"],
            program_name=record["program_name"],
            category=record["category"],
            prompt=record["prompt"],
            target=record["target"],
            split_family=record.get("split_family"),
            input_values=record.get("input_values"),
            instruction=record.get("instruction"),
            num_steps=record.get("num_steps"),
            selection_key=record.get("selection_key"),
        )
        for record in records
    ]


def filter_examples_by_max_steps(examples: List[Example], max_steps: int | None) -> List[Example]:
    if max_steps is None:
        return examples
    return [
        example
        for example in examples
        if example.num_steps is None or example.num_steps <= max_steps
    ]


def canonicalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def exact_match(prediction: str, target: str) -> bool:
    return canonicalize(prediction) == canonicalize(target)


STATE_LINE_PATTERNS = {
    "ip": re.compile(r"^IP=(-?\d+)$"),
    "flags": re.compile(r"^Z=([01]) N=([01]) C=([01])$"),
    "out0": re.compile(r"^OUT\[0\]=(.+)$"),
    "halted": re.compile(r"^HALTED=([01])$"),
    "error": re.compile(r"^ERROR=(.+)$"),
}


def parse_state_text(text: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {"registers": {}}
    for raw_line in canonicalize(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if match := STATE_LINE_PATTERNS["ip"].fullmatch(line):
            parsed["IP"] = int(match.group(1))
            continue
        if line.startswith("R0="):
            for token in line.split():
                name, value = token.split("=", 1)
                parsed["registers"][name] = int(value)
            continue
        if match := STATE_LINE_PATTERNS["flags"].fullmatch(line):
            parsed["Z"] = int(match.group(1))
            parsed["N"] = int(match.group(2))
            parsed["C"] = int(match.group(3))
            continue
        if match := STATE_LINE_PATTERNS["out0"].fullmatch(line):
            parsed["OUT[0]"] = match.group(1)
            continue
        if match := STATE_LINE_PATTERNS["halted"].fullmatch(line):
            parsed["HALTED"] = int(match.group(1))
            continue
        if match := STATE_LINE_PATTERNS["error"].fullmatch(line):
            parsed["ERROR"] = match.group(1)
            continue
    return parsed


def state_field_metrics(prediction: str, target: str) -> Dict[str, object]:
    pred = parse_state_text(prediction)
    gold = parse_state_text(target)
    comparable_fields = ["IP", "Z", "N", "C", "OUT[0]", "HALTED", "ERROR"]
    field_matches: Dict[str, bool] = {}

    for field in comparable_fields:
        if field in gold:
            field_matches[field] = pred.get(field) == gold.get(field)

    gold_registers = gold.get("registers", {})
    pred_registers = pred.get("registers", {})
    register_matches = {
        reg: pred_registers.get(reg) == gold_registers.get(reg)
        for reg in gold_registers
    }

    total_fields = len(field_matches) + len(register_matches)
    matched_fields = sum(field_matches.values()) + sum(register_matches.values())

    return {
        "field_matches": field_matches,
        "register_matches": register_matches,
        "matched_fields": matched_fields,
        "total_fields": total_fields,
        "field_accuracy": (matched_fields / total_fields) if total_fields else 0.0,
    }


def _prompt_section_text(prompt: str, label: str) -> str:
    lines = canonicalize(prompt).splitlines()
    collecting = False
    collected: List[str] = []
    section_labels = {"INPUT", "PROGRAM", "TRACE", "S0", "S1", "S2", "E1", "E2", "TARGET"}
    for raw_line in lines:
        line = raw_line.strip()
        if line == label:
            collecting = True
            collected = []
            continue
        if collecting and line in section_labels:
            break
        if collecting:
            collected.append(raw_line)
    return canonicalize("\n".join(collected)) if collected else ""


def chained_source_state_from_prompt(stage: str, prompt: str) -> str:
    if stage == "next_2_chained_step1":
        return _prompt_section_text(prompt, "S0")
    if stage == "next_2_chained_step2":
        return _prompt_section_text(prompt, "S1")
    return ""


def chained_delta_metrics(prediction: str, target: str, prompt: str, stage: str) -> Dict[str, object]:
    source_text = chained_source_state_from_prompt(stage, prompt)
    source = parse_state_text(source_text)
    pred = parse_state_text(prediction)
    gold = parse_state_text(target)

    comparable_fields = ["IP", "Z", "N", "C", "OUT[0]", "HALTED", "ERROR"]
    changed_field_matches: Dict[str, bool] = {}
    spurious_field_edits: Dict[str, bool] = {}
    changed_register_matches: Dict[str, bool] = {}
    spurious_register_edits: Dict[str, bool] = {}

    for field in comparable_fields:
        if field not in gold:
            continue
        source_value = source.get(field)
        gold_value = gold.get(field)
        pred_value = pred.get(field)
        if source_value != gold_value:
            changed_field_matches[field] = pred_value == gold_value
        elif pred_value != gold_value:
            spurious_field_edits[field] = True

    gold_registers = gold.get("registers", {})
    pred_registers = pred.get("registers", {})
    source_registers = source.get("registers", {})
    for reg, gold_value in gold_registers.items():
        source_value = source_registers.get(reg)
        pred_value = pred_registers.get(reg)
        if source_value != gold_value:
            changed_register_matches[reg] = pred_value == gold_value
        elif pred_value != gold_value:
            spurious_register_edits[reg] = True

    true_positives = sum(changed_field_matches.values()) + sum(changed_register_matches.values())
    false_negatives = (len(changed_field_matches) - sum(changed_field_matches.values())) + (
        len(changed_register_matches) - sum(changed_register_matches.values())
    )
    false_positives = len(spurious_field_edits) + len(spurious_register_edits)
    denominator = true_positives + false_negatives + false_positives

    return {
        "source_state": source_text,
        "changed_field_matches": changed_field_matches,
        "changed_register_matches": changed_register_matches,
        "spurious_field_edits": spurious_field_edits,
        "spurious_register_edits": spurious_register_edits,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "delta_transition_score": (true_positives / denominator) if denominator else 0.0,
        "delta_exact": float(false_negatives == 0 and false_positives == 0 and denominator > 0),
        "field_accuracy": (true_positives / denominator) if denominator else 0.0,
    }


def canonical_state_from_parsed(parsed: Dict[str, object]) -> str:
    registers = parsed.get("registers", {})
    out0 = parsed.get("OUT[0]", "_")
    lines = [
        f"IP={parsed.get('IP', 0)}",
        " ".join(f"R{i}={registers.get(f'R{i}', 0)}" for i in range(8)),
        f"Z={parsed.get('Z', 0)} N={parsed.get('N', 0)} C={parsed.get('C', 0)}",
        f"OUT[0]={out0}",
        f"HALTED={parsed.get('HALTED', 0)}",
        f"ERROR={parsed.get('ERROR', 'NONE')}",
    ]
    return "\n".join(lines)


def repair_state_prediction(prediction: str) -> str:
    parsed = parse_state_text(prediction)
    return canonical_state_from_parsed(parsed)


def parse_labeled_states(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current_label: str | None = None
    for raw_line in canonicalize(text).splitlines():
        line = raw_line.strip()
        if line in {"S1", "S2"}:
            current_label = line
            sections.setdefault(current_label, [])
            continue
        if current_label is not None and line:
            sections[current_label].append(line)
    return {
        label: "\n".join(lines)
        for label, lines in sections.items()
        if lines
    }


SLOT_LINE_PATTERN = re.compile(r"^(S[12])_(IP|REG|FLAGS|OUT|HALTED|ERROR)=(.*)$")
EFFECT_LINE_PATTERN = re.compile(r"^STEP([12])_(OP|DST|SRC|VALUE|IP|FLAGS|OUT|HALTED|ERROR)=(.*)$")
TARGET_ANCHOR_EFFECT_PREFIX = "STEP1_OP="


def parse_slot_labeled_states(text: str) -> Dict[str, Dict[str, object]]:
    sections: Dict[str, Dict[str, object]] = {}
    for raw_line in canonicalize(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SLOT_LINE_PATTERN.fullmatch(line)
        if not match:
            continue
        label, field, value = match.groups()
        parsed = sections.setdefault(label, {"registers": {}})
        value = value.strip()
        if field == "IP":
            parsed["IP"] = int(value)
        elif field == "REG":
            registers: Dict[str, int] = {}
            for token in value.split():
                if "=" not in token:
                    continue
                name, reg_value = token.split("=", 1)
                registers[name] = int(reg_value)
            parsed["registers"] = registers
        elif field == "FLAGS":
            parts = dict(token.split("=", 1) for token in value.split() if "=" in token)
            parsed["Z"] = int(parts.get("Z", 0))
            parsed["N"] = int(parts.get("N", 0))
            parsed["C"] = int(parts.get("C", 0))
        elif field == "OUT":
            parsed["OUT[0]"] = value
        elif field == "HALTED":
            parsed["HALTED"] = int(value)
        elif field == "ERROR":
            parsed["ERROR"] = value
    return sections


def parse_effect_steps(text: str) -> Dict[str, Dict[str, str]]:
    sections: Dict[str, Dict[str, str]] = {}
    for raw_line in canonicalize(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = EFFECT_LINE_PATTERN.fullmatch(line)
        if not match:
            continue
        step_num, field, value = match.groups()
        sections.setdefault(f"STEP{step_num}", {})[field] = value.strip()
    return sections


def canonical_next_2_slots_prediction(prediction: str) -> str:
    sections = parse_slot_labeled_states(prediction)
    parts: List[str] = []
    for label in ("S1", "S2"):
        if label not in sections:
            continue
        parsed = sections[label]
        registers = parsed.get("registers", {})
        parts.append(f"{label}_IP={parsed.get('IP', 0)}")
        parts.append(f"{label}_REG=" + " ".join(f"R{i}={registers.get(f'R{i}', 0)}" for i in range(8)))
        parts.append(f"{label}_FLAGS=Z={parsed.get('Z', 0)} N={parsed.get('N', 0)} C={parsed.get('C', 0)}")
        parts.append(f"{label}_OUT={parsed.get('OUT[0]', '_')}")
        parts.append(f"{label}_HALTED={parsed.get('HALTED', 0)}")
        parts.append(f"{label}_ERROR={parsed.get('ERROR', 'NONE')}")
    return "\n".join(parts).strip()


def canonical_next_2_effect_prediction(prediction: str) -> str:
    sections = parse_effect_steps(prediction)
    parts: List[str] = []
    ordered_fields = ("OP", "DST", "SRC", "VALUE", "IP", "FLAGS", "OUT", "HALTED", "ERROR")
    for label in ("STEP1", "STEP2"):
        fields = sections.get(label)
        if not fields:
            continue
        for field in ordered_fields:
            parts.append(f"{label}_{field}={fields.get(field, 'NONE')}")
    return "\n".join(parts).strip()


def canonical_next_2_effect_target_anchor_prediction(prediction: str) -> str:
    return canonical_next_2_effect_prediction(f"{TARGET_ANCHOR_EFFECT_PREFIX}{prediction}")


def slots_to_next_2_canonical(text: str) -> str:
    sections = parse_slot_labeled_states(text)
    parts: List[str] = []
    for label in ("S1", "S2"):
        if label not in sections:
            continue
        parts.extend([label, canonical_state_from_parsed(sections[label])])
        parts.append("")
    return "\n".join(parts).strip()


def canonical_next_2_prediction(prediction: str) -> str:
    sections = parse_labeled_states(prediction)
    parts: List[str] = []
    for label in ("S1", "S2"):
        if label in sections:
            parts.extend([label, canonical_state_from_parsed(parse_state_text(sections[label]))])
            parts.append("")
    return "\n".join(parts).strip()


def next_2_field_metrics(prediction: str, target: str) -> Dict[str, object]:
    pred_sections = parse_labeled_states(prediction)
    gold_sections = parse_labeled_states(target)
    per_state: Dict[str, Dict[str, object]] = {}
    accuracies: List[float] = []
    for label in ("S1", "S2"):
        if label in gold_sections:
            if label not in pred_sections:
                gold_parsed = parse_state_text(gold_sections[label])
                zero_metrics = {
                    "field_matches": {field: False for field in ["IP", "Z", "N", "C", "OUT[0]", "HALTED", "ERROR"] if field in gold_parsed},
                    "register_matches": {
                        reg: False for reg in gold_parsed.get("registers", {})
                    },
                    "matched_fields": 0,
                    "total_fields": 7 + len(gold_parsed.get("registers", {})),
                    "field_accuracy": 0.0,
                }
                metrics = zero_metrics
            else:
                metrics = state_field_metrics(
                    canonical_state_from_parsed(parse_state_text(pred_sections[label])),
                    canonical_state_from_parsed(parse_state_text(gold_sections[label])),
                )
            per_state[label] = metrics
            accuracies.append(float(metrics["field_accuracy"]))
    return {
        "per_state": per_state,
        "field_accuracy": (sum(accuracies) / len(accuracies)) if accuracies else 0.0,
    }


def next_2_slots_field_metrics(prediction: str, target: str) -> Dict[str, object]:
    return next_2_field_metrics(
        slots_to_next_2_canonical(prediction),
        slots_to_next_2_canonical(target),
    )


def next_2_effect_field_metrics(prediction: str, target: str) -> Dict[str, object]:
    pred_steps = parse_effect_steps(prediction)
    gold_steps = parse_effect_steps(target)
    per_step: Dict[str, Dict[str, object]] = {}
    accuracies: List[float] = []
    ordered_fields = ("OP", "DST", "SRC", "VALUE", "IP", "FLAGS", "OUT", "HALTED", "ERROR")
    for label in ("STEP1", "STEP2"):
        if label not in gold_steps:
            continue
        gold_fields = gold_steps[label]
        pred_fields = pred_steps.get(label, {})
        field_matches = {
            field: pred_fields.get(field) == gold_fields.get(field)
            for field in ordered_fields
            if field in gold_fields
        }
        total_fields = len(field_matches)
        matched_fields = sum(field_matches.values())
        metrics = {
            "field_matches": field_matches,
            "matched_fields": matched_fields,
            "total_fields": total_fields,
            "field_accuracy": (matched_fields / total_fields) if total_fields else 0.0,
        }
        per_step[label] = metrics
        accuracies.append(float(metrics["field_accuracy"]))
    return {
        "per_step": per_step,
        "field_accuracy": (sum(accuracies) / len(accuracies)) if accuracies else 0.0,
    }


def repair_prediction_for_stage(stage: str, prediction: str) -> str:
    if stage in {"single_step", "terminal_state", "next_2_chained_step1", "next_2_chained_step2"}:
        return repair_state_prediction(prediction)
    if stage == "next_2_steps":
        return canonical_next_2_prediction(prediction)
    if stage == "next_2_steps_slots":
        return canonical_next_2_slots_prediction(prediction)
    if stage == "next_2_effects":
        return canonical_next_2_effect_prediction(prediction)
    if stage == "next_2_effects_target_anchor":
        return canonical_next_2_effect_target_anchor_prediction(prediction)
    return canonicalize(prediction)


def stage_system_prompt(stage: str) -> str:
    common = (
        "AI-assembly VM executor.\n"
        "Return canonical target text only.\n"
        "No prose. No markdown. No extra lines."
    )
    if stage == "single_step":
        return common + "\nNext canonical machine state only."
    if stage == "next_2_steps":
        return (
            common
            + "\nEmit S1 and S2 only. S1 is after E1. S2 is after E2. READ updates its destination register. OUT[0] changes only on WRITE."
            + "\nExample:"
            + "\nIN\ninput[0]=5\ninput[1]=7"
            + "\nS0\nIP=0\nR0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
            + "\nE1\nREAD R1, input[0]"
            + "\nE2\nREAD R2, input[1]"
            + "\nS1\nIP=1\nR0=0 R1=5 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
            + "\nS2\nIP=2\nR0=0 R1=5 R2=7 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
        )
    if stage == "next_2_steps_slots":
        return (
            common
            + "\nEmit only these labels in this exact order: S1_IP, S1_REG, S1_FLAGS, S1_OUT, S1_HALTED, S1_ERROR, S2_IP, S2_REG, S2_FLAGS, S2_OUT, S2_HALTED, S2_ERROR."
            + "\nDo not emit instruction text. Do not emit S1 or S2 blocks."
            + "\nExample:"
            + "\nIN\ninput[0]=5\ninput[1]=7"
            + "\nS0\nIP=0\nR0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
            + "\nE1\nREAD R1, input[0]"
            + "\nE2\nREAD R2, input[1]"
            + "\nS1_IP=1"
            + "\nS1_REG=R0=0 R1=5 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0"
            + "\nS1_FLAGS=Z=0 N=0 C=0"
            + "\nS1_OUT=_"
            + "\nS1_HALTED=0"
            + "\nS1_ERROR=NONE"
            + "\nS2_IP=2"
            + "\nS2_REG=R0=0 R1=5 R2=7 R3=0 R4=0 R5=0 R6=0 R7=0"
            + "\nS2_FLAGS=Z=0 N=0 C=0"
            + "\nS2_OUT=_"
            + "\nS2_HALTED=0"
            + "\nS2_ERROR=NONE"
        )
    if stage == "next_2_effects":
        return (
            common
            + "\nEmit only these labels in this exact order: STEP1_OP, STEP1_DST, STEP1_SRC, STEP1_VALUE, STEP1_IP, STEP1_FLAGS, STEP1_OUT, STEP1_HALTED, STEP1_ERROR, STEP2_OP, STEP2_DST, STEP2_SRC, STEP2_VALUE, STEP2_IP, STEP2_FLAGS, STEP2_OUT, STEP2_HALTED, STEP2_ERROR."
            + "\nDescribe the effect of each step, not the whole machine state."
            + "\nExample:"
            + "\nIN\ninput[0]=5\ninput[1]=7"
            + "\nS0\nIP=0\nR0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
            + "\nE1\nREAD R1, input[0]"
            + "\nE2\nREAD R2, input[1]"
            + "\nSTEP1_OP=READ"
            + "\nSTEP1_DST=R1"
            + "\nSTEP1_SRC=input[0]"
            + "\nSTEP1_VALUE=5"
            + "\nSTEP1_IP=1"
            + "\nSTEP1_FLAGS=Z=0 N=0 C=0"
            + "\nSTEP1_OUT=_"
            + "\nSTEP1_HALTED=0"
            + "\nSTEP1_ERROR=NONE"
            + "\nSTEP2_OP=READ"
            + "\nSTEP2_DST=R2"
            + "\nSTEP2_SRC=input[1]"
            + "\nSTEP2_VALUE=7"
            + "\nSTEP2_IP=2"
            + "\nSTEP2_FLAGS=Z=0 N=0 C=0"
            + "\nSTEP2_OUT=_"
            + "\nSTEP2_HALTED=0"
            + "\nSTEP2_ERROR=NONE"
        )
    if stage == "short_trace":
        return (
            common
            + "\nFull state trace only."
            + "\nExample:"
            + "\nPROGRAM\nCONST R1, 4\nHALT\nINPUT\n<none>"
            + "\nTRACE"
            + "\nS1\nIP=1\nR0=0 R1=4 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
            + "\n\nS2\nIP=1\nR0=0 R1=4 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=1\nERROR=NONE"
        )
    if stage == "terminal_state":
        return common + "\nFinal canonical machine state only."
    return common


def _instruction_opcode(instruction: str | None) -> str | None:
    if not instruction:
        return None
    return instruction.split(None, 1)[0]


def _instruction_signature(instruction: str | None) -> str | None:
    if not instruction:
        return None
    opcode, *rest = instruction.split(None, 1)
    operands_text = rest[0] if rest else ""
    normalized_operands = []
    for operand in [part.strip() for part in operands_text.split(",") if part.strip()]:
        if operand.startswith("R"):
            normalized_operands.append("reg")
        elif operand.startswith("[") and operand.endswith("]"):
            normalized_operands.append("mem")
        elif operand.startswith("input["):
            normalized_operands.append("input")
        elif operand.startswith("output["):
            normalized_operands.append("output")
        else:
            normalized_operands.append("atom")
    return f"{opcode}({'/'.join(normalized_operands)})"


def select_few_shots(stage: str, train_examples: List[Example], query: Example, train_shots: int) -> List[Example]:
    def score(example: Example) -> tuple[int, int, int]:
        same_category = 1 if example.category == query.category else 0
        same_program = 1 if example.program_name == query.program_name else 0
        same_opcode = 0
        same_signature = 0
        if stage == "single_step":
            same_opcode = 1 if _instruction_opcode(example.instruction) == _instruction_opcode(query.instruction) else 0
            same_signature = 1 if _instruction_signature(example.instruction) == _instruction_signature(query.instruction) else 0
            return (same_signature, same_opcode, same_category, same_program)
        if stage == "next_2_steps":
            same_key = 1 if example.selection_key and example.selection_key == query.selection_key else 0
            return (same_key, same_category, same_program)
        return (same_category, same_opcode, same_program)

    ranked = sorted(train_examples, key=score, reverse=True)
    selected: List[Example] = []
    used_programs = set()
    for example in ranked:
        if len(selected) >= train_shots:
            break
        # Prefer diversity across programs when possible.
        if stage != "single_step" and example.program_name in used_programs and len(ranked) > train_shots:
            continue
        selected.append(example)
        used_programs.add(example.program_name)

    if len(selected) < train_shots:
        for example in ranked:
            if len(selected) >= train_shots:
                break
            if example not in selected:
                selected.append(example)
    return selected


def build_few_shot_prompt(stage: str, examples: List[Example], query: Example) -> str:
    if stage == "next_2_steps":
        blocks: List[str] = []
        for example in examples:
            blocks.append(f"X\n{example.prompt}\n=\n{example.target}")
        blocks.append(f"Q\n{query.prompt}\n=")
        return "\n\n".join(blocks)

    blocks: List[str] = []
    for index, example in enumerate(examples, start=1):
        blocks.append(f"EXAMPLE {index}\n{example.prompt}\nTARGET\n{example.target}")
    blocks.append(f"QUERY\n{query.prompt}\nTARGET")
    return "\n\n".join(blocks)


def _example_cost(example: Example) -> int:
    return len(example.prompt) + len(example.target)


def evaluate_stage(
    client: OllamaClient,
    dataset_root: Path,
    stage: str,
    train_shots: int,
    eval_limit: int,
    max_trace_steps: int | None,
) -> dict:
    effective_max_trace_steps = max_trace_steps
    if stage == "short_trace":
        if effective_max_trace_steps is None:
            effective_max_trace_steps = DEFAULT_SHORT_TRACE_STEP_CAP
        else:
            effective_max_trace_steps = min(effective_max_trace_steps, DEFAULT_SHORT_TRACE_STEP_CAP)

    train_examples = filter_examples_by_max_steps(load_split(dataset_root, stage, "train"), effective_max_trace_steps)
    val_examples = filter_examples_by_max_steps(load_split(dataset_root, stage, "val"), effective_max_trace_steps)
    test_examples = filter_examples_by_max_steps(load_split(dataset_root, stage, "test"), effective_max_trace_steps)
    system_prompt = stage_system_prompt(stage)

    def run_eval(split_name: str, examples: List[Example]) -> dict:
        selected = sorted(examples, key=_example_cost)[: min(eval_limit, len(examples))]
        results: List[dict] = []
        field_accuracy_sum = 0.0
        for index, example in enumerate(selected, start=1):
            few_shots = select_few_shots(stage, train_examples, example, min(train_shots, len(train_examples)))
            prompt = build_few_shot_prompt(stage, few_shots, example)
            started = time.time()
            try:
                prediction = client.generate(prompt=prompt, system=system_prompt)
            except RuntimeError as exc:
                prediction = f"ERROR=REQUEST_TIMEOUT_OR_FAILURE: {exc}"
            elapsed_ms = int((time.time() - started) * 1000)
            repaired_prediction = repair_prediction_for_stage(stage, prediction)
            if stage in {"single_step", "terminal_state"}:
                relaxed_metrics = state_field_metrics(repaired_prediction, example.target)
            elif stage in {"next_2_chained_step1", "next_2_chained_step2"}:
                relaxed_metrics = chained_delta_metrics(repaired_prediction, example.target, example.prompt, stage)
            elif stage == "next_2_steps":
                relaxed_metrics = next_2_field_metrics(repaired_prediction, example.target)
            elif stage == "next_2_steps_slots":
                relaxed_metrics = next_2_slots_field_metrics(repaired_prediction, example.target)
            elif stage == "next_2_effects":
                relaxed_metrics = next_2_effect_field_metrics(repaired_prediction, example.target)
            else:
                relaxed_metrics = {}
            field_accuracy_sum += relaxed_metrics.get("field_accuracy", 0.0)
            results.append(
                {
                    "index": index,
                    "program_name": example.program_name,
                    "category": example.category,
                    "split_family": example.split_family,
                    "exact_match": exact_match(prediction, example.target),
                    "repaired_exact_match": exact_match(repaired_prediction, example.target),
                    "field_accuracy": relaxed_metrics.get("field_accuracy"),
                    "field_matches": relaxed_metrics.get("field_matches"),
                    "register_matches": relaxed_metrics.get("register_matches"),
                    "changed_field_matches": relaxed_metrics.get("changed_field_matches"),
                    "changed_register_matches": relaxed_metrics.get("changed_register_matches"),
                    "spurious_field_edits": relaxed_metrics.get("spurious_field_edits"),
                    "spurious_register_edits": relaxed_metrics.get("spurious_register_edits"),
                    "delta_transition_score": relaxed_metrics.get("delta_transition_score"),
                    "delta_exact": relaxed_metrics.get("delta_exact"),
                    "per_state_metrics": relaxed_metrics.get("per_state"),
                    "elapsed_ms": elapsed_ms,
                    "few_shot_programs": [shot.program_name for shot in few_shots],
                    "prediction": prediction,
                    "repaired_prediction": repaired_prediction,
                    "target": example.target,
                }
            )
        exact = sum(1 for item in results if item["exact_match"])
        repaired_exact = sum(1 for item in results if item["repaired_exact_match"])
        return {
            "count": len(results),
            "exact_match_count": exact,
            "exact_match_rate": (exact / len(results)) if results else 0.0,
            "repaired_exact_match_count": repaired_exact,
            "repaired_exact_match_rate": (repaired_exact / len(results)) if results else 0.0,
            "avg_field_accuracy": (field_accuracy_sum / len(results)) if results and stage in {"single_step", "terminal_state", "next_2_steps", "next_2_steps_slots", "next_2_effects", "next_2_chained_step1", "next_2_chained_step2"} else None,
            "results": results,
        }

    return {
        "stage": stage,
        "train_shots": min(train_shots, len(train_examples)),
        "max_trace_steps": effective_max_trace_steps,
        "train_source_count": len(train_examples),
        "system_prompt": system_prompt,
        "val": run_eval("val", val_examples),
        "test": run_eval("test", test_examples),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Ollama baseline trainer/evaluator over AI-assembly datasets.")
    parser.add_argument("--dataset-root", default="datasets/mvp")
    parser.add_argument("--report-dir", default="reports/baseline")
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--train-shots", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=192)
    parser.add_argument("--max-trace-steps", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_HTTP_TIMEOUT_SECONDS)
    parser.add_argument("--stages", nargs="*", default=DEFAULT_STAGES)
    args = parser.parse_args()

    available, details = check_ollama_available(args.host, min(args.timeout_seconds, 5))
    if not available:
        raise SystemExit(f"Ollama runtime is unavailable at {args.host}: {details}")

    dataset_root = Path(args.dataset_root)
    report_dir = Path(args.report_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_slug = args.model.replace(":", "-").replace("/", "-")
    run_id = f"{timestamp}-{model_slug}-{os.getpid()}"
    run_dir = report_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "run_config.json",
        {
            "model": args.model,
            "host": args.host,
            "train_shots": args.train_shots,
            "eval_limit": args.eval_limit,
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "max_trace_steps": args.max_trace_steps,
            "timeout_seconds": args.timeout_seconds,
            "stages": args.stages,
        },
    )

    stage_reports: Dict[str, dict] = {}
    for stage in args.stages:
        num_predict = args.num_predict
        if stage == "single_step":
            num_predict = min(num_predict, 192)
        elif stage == "next_2_steps":
            num_predict = min(num_predict, 256)
        elif stage == "short_trace":
            num_predict = min(num_predict, 384)
        elif stage == "terminal_state":
            num_predict = min(num_predict, 256)
        client = OllamaClient(
            host=args.host,
            model=args.model,
            temperature=args.temperature,
            num_predict=num_predict,
            timeout_seconds=args.timeout_seconds,
        )
        stage_reports[stage] = evaluate_stage(
            client=client,
            dataset_root=dataset_root,
            stage=stage,
            train_shots=args.train_shots,
            eval_limit=args.eval_limit,
            max_trace_steps=args.max_trace_steps,
        )
        write_json(run_dir / f"{stage}.json", stage_reports[stage])
        write_json(
            run_dir / "summary.json",
            {
                "model": args.model,
                "host": args.host,
                "train_shots": args.train_shots,
                "eval_limit": args.eval_limit,
                "temperature": args.temperature,
                "num_predict": args.num_predict,
                "max_trace_steps": args.max_trace_steps,
                "timeout_seconds": args.timeout_seconds,
                "completed_stages": list(stage_reports.keys()),
                "stages": {
                    done_stage: {
                        "val_exact_match_rate": report["val"]["exact_match_rate"],
                        "test_exact_match_rate": report["test"]["exact_match_rate"],
                        "val_repaired_exact_match_rate": report["val"]["repaired_exact_match_rate"],
                        "test_repaired_exact_match_rate": report["test"]["repaired_exact_match_rate"],
                        "val_avg_field_accuracy": report["val"]["avg_field_accuracy"],
                        "test_avg_field_accuracy": report["test"]["avg_field_accuracy"],
                        "val_count": report["val"]["count"],
                        "test_count": report["test"]["count"],
                    }
                    for done_stage, report in stage_reports.items()
                },
            },
        )

    summary = {
        "model": args.model,
        "host": args.host,
        "train_shots": args.train_shots,
        "eval_limit": args.eval_limit,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "max_trace_steps": args.max_trace_steps,
        "timeout_seconds": args.timeout_seconds,
        "stages": {
            stage: {
                "val_exact_match_rate": report["val"]["exact_match_rate"],
                "test_exact_match_rate": report["test"]["exact_match_rate"],
                "val_repaired_exact_match_rate": report["val"]["repaired_exact_match_rate"],
                "test_repaired_exact_match_rate": report["test"]["repaired_exact_match_rate"],
                "val_avg_field_accuracy": report["val"]["avg_field_accuracy"],
                "test_avg_field_accuracy": report["test"]["avg_field_accuracy"],
                "val_count": report["val"]["count"],
                "test_count": report["test"]["count"],
            }
            for stage, report in stage_reports.items()
        },
    }

    write_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
