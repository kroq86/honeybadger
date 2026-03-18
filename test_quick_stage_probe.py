import unittest

from training.quick_stage_probe import _score_field_accuracy


class QuickStageProbeTests(unittest.TestCase):
    def test_scores_chained_step_raises_without_prompt(self) -> None:
        target = "IP=1\nR0=0 R1=9 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
        with self.assertRaises(ValueError):
            _score_field_accuracy("next_2_chained_step1", target, target)

    def test_scores_next2_effects_exact_match(self) -> None:
        target = (
            "STEP1_OP=READ\nSTEP1_DST=R1\nSTEP1_SRC=input[4]\nSTEP1_VALUE=9\nSTEP1_IP=1\nSTEP1_FLAGS=Z=0 N=0 C=0\nSTEP1_OUT=_\nSTEP1_HALTED=0\nSTEP1_ERROR=NONE\n"
            "STEP2_OP=READ\nSTEP2_DST=R2\nSTEP2_SRC=input[0]\nSTEP2_VALUE=2\nSTEP2_IP=2\nSTEP2_FLAGS=Z=0 N=0 C=0\nSTEP2_OUT=_\nSTEP2_HALTED=0\nSTEP2_ERROR=NONE"
        )
        self.assertEqual(_score_field_accuracy("next_2_effects", target, target), 1.0)


if __name__ == "__main__":
    unittest.main()
