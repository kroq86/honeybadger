from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_pipeline import TASK_LIBRARY
from reference_vm import collect_transitions


def build_two_step_windows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in TASK_LIBRARY:
        for case_index, input_values in enumerate(task.input_cases):
            _, transitions = collect_transitions(task.source, inputs=input_values)
            for window_start in range(len(transitions) - 1):
                first = transitions[window_start]
                second = transitions[window_start + 1]
                rows.append(
                    {
                        "dataset_type": "next_2_steps_search_benchmark",
                        "program_name": task.program_name,
                        "case_index": case_index,
                        "input_values": input_values,
                        "window_start": window_start,
                        "prompt": "\n".join(
                            [
                                "TASK: next_2_steps_search_benchmark",
                                "Emit S2 only.",
                                "INPUT",
                                *[f"input[{key}]={value}" for key, value in sorted(input_values.items())],
                                "S0",
                                first.before_state_text,
                            ]
                        ),
                        "target": "\n".join(
                            [
                                "S1",
                                first.after_state_text,
                                "",
                                "S2",
                                second.after_state_text,
                            ]
                        ),
                        "gold_path": [first.instruction, second.instruction],
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build two-step search benchmark from VM traces.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_two_step_windows()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
