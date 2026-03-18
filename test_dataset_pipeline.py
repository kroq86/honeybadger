import json
import tempfile
import unittest
from pathlib import Path

from dataset_pipeline import (
    build_manifest,
    build_dataset_card,
    build_generation_config,
    generate_next_k_steps_examples,
    generate_short_trace_examples,
    generate_single_step_examples,
    generate_terminal_state_examples,
    split_records,
    write_generation_config,
    write_jsonl,
    write_manifest,
)


class DatasetPipelineTests(unittest.TestCase):
    def test_single_step_examples_have_prompt_and_target(self) -> None:
        examples = generate_single_step_examples(limit=8, seed=1)

        self.assertTrue(examples)
        sample = examples[0]
        self.assertEqual(sample["dataset_type"], "single_step")
        self.assertIn("TASK: single_step_execution", sample["prompt"])
        self.assertIn("EXEC", sample["prompt"])
        self.assertIn("IP=", sample["target"])

    def test_short_trace_examples_have_multi_step_target(self) -> None:
        examples = generate_short_trace_examples(limit=4, seed=1)

        self.assertTrue(examples)
        sample = examples[0]
        self.assertEqual(sample["dataset_type"], "short_trace")
        self.assertIn("TASK: short_trace_execution", sample["prompt"])
        self.assertIn("S1", sample["target"])
        self.assertIn("num_steps", sample)

    def test_next_k_steps_examples_have_prefix_target(self) -> None:
        examples = generate_next_k_steps_examples(limit=4, seed=1, next_k_steps=2, dataset_type="next_2_steps")

        self.assertTrue(examples)
        sample = examples[0]
        self.assertEqual(sample["dataset_type"], "next_2_steps")
        self.assertIn("TASK: next_2_steps_execution", sample["prompt"])
        self.assertIn("Emit S1 and S2 only.", sample["prompt"])
        self.assertIn("S0", sample["prompt"])
        self.assertIn("E1", sample["prompt"])
        self.assertIn("E2", sample["prompt"])
        self.assertIn("S1", sample["target"])
        self.assertIn("S2", sample["target"])
        self.assertEqual(sample["num_steps"], 2)

    def test_terminal_state_examples_have_final_state_target(self) -> None:
        examples = generate_terminal_state_examples(limit=4, seed=1)

        self.assertTrue(examples)
        sample = examples[0]
        self.assertEqual(sample["dataset_type"], "terminal_state")
        self.assertIn("TASK: terminal_state_execution", sample["prompt"])
        self.assertIn("Emit final canonical state only.", sample["prompt"])
        self.assertIn("HALTED=1", sample["target"])

    def test_split_records_creates_train_val_test(self) -> None:
        records = [
            {"id": 1, "split_family": "fam_a"},
            {"id": 2, "split_family": "fam_a"},
            {"id": 3, "split_family": "fam_b"},
            {"id": 4, "split_family": "fam_c"},
            {"id": 5, "split_family": "fam_d"},
        ]
        splits = split_records(records, seed=1)

        self.assertEqual(set(splits.keys()), {"train", "val", "test"})
        self.assertEqual(sum(len(records) for records in splits.values()), len(records))
        family_to_split = {}
        for split_name, split_records_list in splits.items():
            for record in split_records_list:
                family = record["split_family"]
                if family in family_to_split:
                    self.assertEqual(family_to_split[family], split_name)
                family_to_split[family] = split_name

    def test_manifest_contains_distribution_stats(self) -> None:
        generated = {
            "single_step": split_records(generate_single_step_examples(limit=12, seed=1), seed=1),
            "next_2_steps": split_records(
                generate_next_k_steps_examples(limit=6, seed=1, next_k_steps=2, dataset_type="next_2_steps"),
                seed=1,
            ),
            "short_trace": split_records(generate_short_trace_examples(limit=6, seed=1), seed=1),
            "terminal_state": split_records(generate_terminal_state_examples(limit=6, seed=1), seed=1),
        }
        manifest = build_manifest(generated, seed=1, output_dir=Path("/tmp/demo"))

        self.assertIn("overall_counts", manifest)
        self.assertIn("split_counts", manifest)
        self.assertIn("category_distribution", manifest)
        self.assertIn("program_distribution", manifest)
        self.assertIn("split_family_distribution", manifest)

    def test_write_jsonl_persists_records(self) -> None:
        records = [{"a": 1}, {"b": 2}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "records.jsonl"
            write_jsonl(path, records)

            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0]), {"a": 1})

    def test_write_manifest_persists_json(self) -> None:
        manifest = {"ok": True, "count": 3}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.json"
            write_manifest(path, manifest)

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["count"], 3)

    def test_generation_config_and_dataset_card(self) -> None:
        class Args:
            seed = 7
            output_dir = "/tmp/demo"
            single_step_limit = 10
            next_2_steps_limit = 5
            short_trace_limit = 5
            terminal_state_limit = 5

        config = build_generation_config(Args())
        manifest = {
            "dataset_types": ["single_step", "next_2_steps", "short_trace", "terminal_state"],
            "overall_counts": {
                "single_step": 10,
                "next_2_steps": 5,
                "short_trace": 5,
                "terminal_state": 5,
            },
            "split_counts": {"single_step": {"train": 7, "val": 1, "test": 2}},
        }
        card = build_dataset_card(manifest, config)

        self.assertIn("program family level", card)
        self.assertEqual(config["split_strategy"], "program_family_level")
        self.assertEqual(config["dataset_types"], ["single_step", "next_2_steps", "short_trace", "terminal_state"])

    def test_write_generation_config_persists_json(self) -> None:
        config = {"seed": 7, "split_strategy": "program_family_level"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "generation_config.json"
            write_generation_config(path, config)

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["split_strategy"], "program_family_level")


if __name__ == "__main__":
    unittest.main()
