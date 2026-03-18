from __future__ import annotations

import unittest

from vm_transition_verifier import verify_single_step, verification_mode_verdict


class VerificationModesTests(unittest.TestCase):
    def test_instruction_only_rejects_wrong_instruction(self) -> None:
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
        ok, _ = verification_mode_verdict(result, mode="instruction_only")
        self.assertFalse(ok)

    def test_state_diff_accepts_matching_effects(self) -> None:
        before_state_text = "\n".join(
            [
                "IP=1",
                "R0=0 R1=5 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0",
                "Z=0 N=0 C=0",
                "OUT[0]=_",
                "HALTED=0",
                "ERROR=NONE",
            ]
        )
        target_state_text = "\n".join(
            [
                "IP=2",
                "R0=0 R1=5 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0",
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
            target_state_text=target_state_text,
        )
        ok, _ = verification_mode_verdict(result, mode="state_diff")
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
