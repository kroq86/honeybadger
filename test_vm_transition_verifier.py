from __future__ import annotations

import unittest

from test_paths import repo_path
from reference_vm import MachineState
from vm_transition_verifier import (
    load_jsonl,
    parse_state_text,
    replay_dataset,
    verify_single_step,
    verify_next_2_steps_record,
    verify_single_step_record,
)


class VmTransitionVerifierTest(unittest.TestCase):
    def test_parse_state_text_roundtrip(self) -> None:
        state = MachineState()
        state.ip = 3
        state.registers[1] = 7
        state.registers[4] = -2
        state.z = 0
        state.n = 1
        state.c = 0
        state.output[0] = 9
        state.memory[10] = 42
        parsed = parse_state_text(state.serialize())
        self.assertEqual(parsed.serialize(), state.serialize())

    def test_single_step_replay_matrix_seed11(self) -> None:
        path = repo_path("datasets", "matrix_seed11", "single_step", "test.jsonl")
        records = load_jsonl(path)
        self.assertGreater(len(records), 0)
        failures = [
            record["program_name"]
            for record in records
            if not verify_single_step_record(record).valid
        ]
        self.assertEqual(failures, [])

    def test_next_2_steps_replay_matrix_seed11(self) -> None:
        path = repo_path("datasets", "matrix_seed11", "next_2_steps", "test.jsonl")
        records = load_jsonl(path)
        self.assertGreater(len(records), 0)
        failures = []
        for record in records:
            results = verify_next_2_steps_record(record)
            if not all(result.valid for result in results):
                failures.append(
                    {
                        "program_name": record["program_name"],
                        "steps": [
                            {
                                "expected": result.expected_instruction,
                                "actual": result.actual_instruction,
                                "notes": list(result.notes),
                            }
                            for result in results
                        ],
                    }
                )
        self.assertEqual(failures, [])

    def test_replay_dataset_supports_single_and_next2(self) -> None:
        single_path = repo_path(
            "datasets", "matrix_seed11", "single_step", "test.jsonl"
        )
        next2_path = repo_path(
            "datasets", "matrix_seed11", "next_2_steps", "test.jsonl"
        )
        single_rows = replay_dataset(single_path)
        next2_rows = replay_dataset(next2_path)
        self.assertTrue(all(row["valid"] for row in single_rows))
        self.assertTrue(all(row["valid"] for row in next2_rows))

    def test_wrong_candidate_instruction_is_executed_not_expected_ip_instruction(
        self,
    ) -> None:
        before_state_text = "\n".join(
            [
                "IP=0",
                "R0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0",
                "Z=0 N=0 C=0",
                "OUT[0]=_",
                "HALTED=0",
                "ERROR=NONE",
            ]
        )
        result = verify_single_step(
            program_name="countdown_to_zero",
            input_values={0: 5},
            before_state_text=before_state_text,
            instruction_text="CONST R2, 0",
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.expected_instruction, "READ R1, input[0]")
        self.assertIn("R1=0", result.after_state_text)


if __name__ == "__main__":
    unittest.main()
