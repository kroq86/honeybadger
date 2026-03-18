import unittest

from training.next2_effect_target_anchor_export import convert_benchmark_record_to_anchor


class Next2EffectTargetAnchorExportTests(unittest.TestCase):
    def test_convert_benchmark_record_to_anchor_updates_prompt_and_target(self) -> None:
        record = {
            "dataset_type": "next_2_steps",
            "program_name": "demo",
            "prompt": (
                "TASK: next_2_steps_execution\nEmit S1 and S2 only.\n"
                "INPUT\ninput[0]=2\nS0\nIP=0\nR0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\n"
                "Z=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE\nE1\nREAD R1, input[0]\nE2\nCONST R2, 0"
            ),
            "target": (
                "S1\nIP=1\nR0=0 R1=2 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE\n\n"
                "S2\nIP=2\nR0=0 R1=2 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
            ),
        }
        converted = convert_benchmark_record_to_anchor(record)
        self.assertEqual(converted["dataset_type"], "next_2_effects_target_anchor")
        self.assertTrue(converted["prompt"].endswith("TARGET\nSTEP1_OP="))
        self.assertFalse(converted["target"].startswith("STEP1_OP="))
        self.assertTrue(converted["target"].startswith("READ"))


if __name__ == "__main__":
    unittest.main()
