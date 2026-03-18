import unittest

from repair_dataset import generate_repair_examples, validate_repair_target


class RepairDatasetTests(unittest.TestCase):
    def test_generate_examples_have_canonical_prompt_shape(self) -> None:
        records = generate_repair_examples(limit=4, seed=1)
        self.assertTrue(records)
        sample = records[0]
        self.assertEqual(sample["dataset_type"], "program_repair")
        self.assertIn("TASK: program_repair", sample["prompt"])
        self.assertIn("BUGGY_PROGRAM", sample["prompt"])
        self.assertIn("IO_EXAMPLES", sample["prompt"])
        self.assertIn("bug_class=", sample["prompt"])

    def test_buggy_program_is_broken_but_parseable(self) -> None:
        records = generate_repair_examples(limit=8, seed=1)
        self.assertTrue(records)
        for record in records:
            self.assertTrue(record["buggy_validation"]["syntactic_valid"])
            self.assertFalse(record["buggy_validation"]["functional_correct"])

    def test_target_is_functionally_correct(self) -> None:
        records = generate_repair_examples(limit=8, seed=1)
        self.assertTrue(records)
        for record in records:
            result = validate_repair_target(record["target"], record["io_examples"])
            self.assertTrue(result["functional_correct"])


if __name__ == "__main__":
    unittest.main()
