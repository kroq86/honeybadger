from __future__ import annotations

import unittest

from test_paths import repo_path
from candidate_generator import generate_candidates
from vm_transition_verifier import load_jsonl, verify_single_step


class CandidateGeneratorTests(unittest.TestCase):
    def test_single_step_gold_instruction_is_top1(self) -> None:
        rows = load_jsonl(
            repo_path("datasets", "matrix_seed11", "single_step", "test.jsonl")
        )
        for row in rows:
            sections = row["prompt"].split("EXEC")
            before_state_text = sections[0].split("STATE", 1)[1].strip()
            candidates = generate_candidates(
                program_name=row["program_name"],
                before_state_text=before_state_text,
                limit=3,
            )
            self.assertGreaterEqual(len(candidates), 1)
            self.assertEqual(candidates[0].instruction_text, row["instruction"])

    def test_jump_candidates_include_target_and_fallthrough(self) -> None:
        state_text = "\n".join(
            [
                "IP=3",
                "R0=0 R1=2 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0",
                "Z=0 N=0 C=0",
                "OUT[0]=_",
                "HALTED=0",
                "ERROR=NONE",
            ]
        )
        candidates = generate_candidates(
            program_name="countdown_to_zero",
            before_state_text=state_text,
            limit=5,
        )
        texts = [candidate.instruction_text for candidate in candidates]
        self.assertIn("JZ DONE", texts)
        self.assertIn("SUB R1, R1, R3", texts)

    def test_next2_first_step_has_valid_candidate(self) -> None:
        rows = load_jsonl(
            repo_path("datasets", "matrix_seed11", "next_2_steps", "test.jsonl")
        )
        for row in rows:
            prompt_lines = row["prompt"].splitlines()
            before_state_text = "\n".join(
                prompt_lines[
                    prompt_lines.index("S0") + 1 : prompt_lines.index("E1")
                ]
            ).strip()
            target_sections = row["target"].split("S2")
            first_target = target_sections[0].split("S1", 1)[1].strip()
            candidates = generate_candidates(
                program_name=row["program_name"],
                before_state_text=before_state_text,
                limit=5,
            )
            self.assertTrue(
                any(
                    verify_single_step(
                        program_name=row["program_name"],
                        input_values=row.get("input_values"),
                        before_state_text=before_state_text,
                        instruction_text=candidate.instruction_text,
                        target_state_text=first_target,
                    ).valid
                    for candidate in candidates
                )
            )


if __name__ == "__main__":
    unittest.main()
