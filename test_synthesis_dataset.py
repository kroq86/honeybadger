import json
import tempfile
import unittest
from pathlib import Path

from synthesis_dataset import (
    build_synthesis_prompt,
    generate_synthesis_examples,
    validate_synthesis_target,
)


class SynthesisDatasetTests(unittest.TestCase):
    def test_generate_examples_have_canonical_prompt_shape(self) -> None:
        records = generate_synthesis_examples(limit=4, seed=1)
        self.assertTrue(records)
        sample = records[0]
        self.assertEqual(sample["dataset_type"], "program_synthesis")
        self.assertIn("TASK: program_synthesis", sample["prompt"])
        self.assertIn("SPEC", sample["prompt"])
        self.assertIn("IO_EXAMPLES", sample["prompt"])
        self.assertIn("EXPECTED", sample["prompt"])
        self.assertTrue(sample["target"].strip().endswith("HALT"))

    def test_targets_are_executable_and_functionally_correct(self) -> None:
        records = generate_synthesis_examples(limit=8, seed=1)
        for record in records:
            result = validate_synthesis_target(record["target"], record["io_examples"])
            self.assertTrue(result["syntactic_valid"])
            self.assertTrue(result["executable_valid"])
            self.assertTrue(result["functional_correct"])

    def test_validation_rejects_broken_program(self) -> None:
        records = generate_synthesis_examples(limit=1, seed=1)
        broken_target = "CONST R1, 1\nWRITE output[0], R1"
        result = validate_synthesis_target(broken_target, records[0]["io_examples"])
        self.assertFalse(result["functional_correct"])

    def test_prompt_contains_named_goal(self) -> None:
        records = generate_synthesis_examples(limit=8, seed=1)
        prompt_by_name = {record["program_name"]: record["prompt"] for record in records}
        self.assertIn("goal=read two inputs and write their sum to output[0]", prompt_by_name["add_two_numbers"])
        self.assertIn("shape=bounded_search", prompt_by_name["midpoint_search_4"])


if __name__ == "__main__":
    unittest.main()
