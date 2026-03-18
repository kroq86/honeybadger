import unittest

from training.check_env import build_env_report


class TrainingEnvTests(unittest.TestCase):
    def test_env_report_shape(self) -> None:
        report = build_env_report()
        self.assertIn("required_stack", report)
        self.assertIn("required_missing", report)
        self.assertIn("ready_for_local_lora_smoke", report)
        self.assertIn("torch", report["required_stack"])


if __name__ == "__main__":
    unittest.main()
