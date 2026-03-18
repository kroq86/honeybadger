import unittest

from curriculum_gate import evaluate_summary


class CurriculumGateTests(unittest.TestCase):
    def test_next_focus_is_first_missing_or_failed_stage(self) -> None:
        summary = {
            "model": "demo",
            "stages": {
                "single_step": {
                    "val_avg_field_accuracy": 0.90,
                    "test_avg_field_accuracy": 0.90,
                    "val_repaired_exact_match_rate": 0.20,
                    "test_repaired_exact_match_rate": 0.20,
                },
                "next_2_steps": {
                    "val_avg_field_accuracy": 0.70,
                    "test_avg_field_accuracy": 0.60,
                    "val_repaired_exact_match_rate": 0.10,
                    "test_repaired_exact_match_rate": 0.05,
                },
            },
        }
        result = evaluate_summary(summary)
        self.assertEqual(result["next_focus_stage"], "next_2_steps")
        self.assertEqual(result["active_curriculum"], ["single_step", "next_2_steps", "short_trace", "terminal_state"])
        self.assertTrue(result["stages"]["single_step"]["signal"])
        self.assertFalse(result["stages"]["next_2_steps"]["signal"])

    def test_pass_stage_requires_thresholds(self) -> None:
        summary = {
            "model": "demo",
            "stages": {
                "single_step": {
                    "val_avg_field_accuracy": 0.96,
                    "test_avg_field_accuracy": 0.95,
                    "val_repaired_exact_match_rate": 0.72,
                    "test_repaired_exact_match_rate": 0.71,
                },
                "next_2_steps": {
                    "val_avg_field_accuracy": 0.90,
                    "test_avg_field_accuracy": 0.88,
                    "val_repaired_exact_match_rate": 0.60,
                    "test_repaired_exact_match_rate": 0.55,
                },
                "short_trace": {
                    "val_repaired_exact_match_rate": 0.03,
                    "test_repaired_exact_match_rate": 0.02,
                },
            },
        }
        result = evaluate_summary(summary)
        self.assertTrue(result["stages"]["single_step"]["pass"])
        self.assertTrue(result["stages"]["next_2_steps"]["pass"])
        self.assertEqual(result["next_focus_stage"], "short_trace")


if __name__ == "__main__":
    unittest.main()
