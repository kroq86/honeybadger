import unittest

from training.prefix_forced_probe import _prefilled_prompt, _repaired_with_seed, _score


class PrefixForcedProbeTests(unittest.TestCase):
    def test_prefilled_prompt_appends_seed(self) -> None:
        prompt = "TASK\nINPUT\n<none>"
        self.assertEqual(_prefilled_prompt(prompt, "IP="), "TASK\nINPUT\n<none>\nIP=")

    def test_repaired_with_seed_reconstructs_state_prefix(self) -> None:
        repaired = _repaired_with_seed(
            "single_step",
            "1\nR0=0 R1=9 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE",
            "IP=",
        )
        self.assertTrue(repaired.startswith("IP=1"))

    def test_score_uses_delta_for_chained_stage(self) -> None:
        prompt = (
            "TASK: chained_step1\nINPUT\ninput[0]=9\nS0\n"
            "IP=0\nR0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\n"
            "Z=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE\nE1\nREAD R1, input[0]"
        )
        target = (
            "IP=1\nR0=0 R1=9 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\n"
            "Z=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
        )
        zero_like = (
            "IP=0\nR0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\n"
            "Z=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
        )
        scored = _score("next_2_chained_step1", zero_like, target, prompt)
        self.assertEqual(scored["delta_transition_score"], 0.0)
        self.assertEqual(scored["delta_exact"], 0.0)


if __name__ == "__main__":
    unittest.main()
