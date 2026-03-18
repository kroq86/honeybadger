from __future__ import annotations

import unittest

from branch_ranker import rank_candidates
from candidate_generator import generate_candidates


class BranchRankerTests(unittest.TestCase):
    def test_heuristic_moves_current_ip_to_front_in_program_global_mode(self) -> None:
        state_text = "\n".join(
            [
                "IP=4",
                "R0=0 R1=3 R2=0 R3=1 R4=0 R5=0 R6=0 R7=0",
                "Z=0 N=0 C=0",
                "OUT[0]=_",
                "HALTED=0",
                "ERROR=NONE",
            ]
        )
        candidates = generate_candidates(
            program_name="countdown_to_zero",
            before_state_text=state_text,
            mode="program_global",
            limit=8,
        )
        ranked = rank_candidates(candidates, before_state_text=state_text, strategy="heuristic")
        self.assertEqual(ranked[0].instruction_text, "JZ DONE")


if __name__ == "__main__":
    unittest.main()
