import unittest

from training.collect_next2_diagnostics import classify_prediction


class CollectNext2DiagnosticsTests(unittest.TestCase):
    def test_classifies_missing_labels(self) -> None:
        result = classify_prediction("garbage", "", "S1\nIP=1\n\nS2\nIP=2")
        self.assertEqual(result["category"], "missing_labels")

    def test_classifies_empty_output(self) -> None:
        result = classify_prediction("", "", "S1\nIP=1\n\nS2\nIP=2")
        self.assertEqual(result["category"], "empty_output")


if __name__ == "__main__":
    unittest.main()
