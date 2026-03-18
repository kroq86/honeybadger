import unittest

from latent_probe_dataset import generate_latent_probe_examples, validate_probe_target
from dataset_pipeline import TASK_LIBRARY


class LatentProbeDatasetTests(unittest.TestCase):
    def test_generate_examples_have_terminal_only_shape(self) -> None:
        records = generate_latent_probe_examples(limit=4, seed=1)
        self.assertTrue(records)
        sample = records[0]
        self.assertEqual(sample["dataset_type"], "latent_terminal_probe")
        self.assertIn("TASK: latent_terminal_probe", sample["prompt"])
        self.assertIn("compression_goal=final_state_without_explicit_trace", sample["prompt"])
        self.assertIn("HALTED=", sample["target"])

    def test_targets_validate_against_reference_vm(self) -> None:
        records = generate_latent_probe_examples(limit=8, seed=1)
        source_by_name = {task.program_name: task.source for task in TASK_LIBRARY}
        for record in records:
            self.assertTrue(
                validate_probe_target(
                    source_by_name[record["program_name"]],
                    record["input_values"],
                    record["target"],
                )
            )


if __name__ == "__main__":
    unittest.main()
