import json
import unittest

from test_paths import repo_path
from training.next2_chained_probe import (
    build_step1_prompt,
    build_step2_prompt,
    split_prompt,
    split_target,
)


class Next2ChainedProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        sample = json.loads(
            repo_path("datasets", "mvp", "next_2_steps", "train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.prompt = sample["prompt"]
        self.target = sample["target"]

    def test_split_prompt_extracts_prefix_and_steps(self) -> None:
        prefix, e1, e2 = split_prompt(self.prompt)
        self.assertIn("S0", prefix)
        self.assertNotIn("TASK: next_2_steps_execution", prefix)
        self.assertTrue(e1)
        self.assertTrue(e2)

    def test_split_target_extracts_states(self) -> None:
        s1, s2 = split_target(self.target)
        self.assertIn("IP=", s1)
        self.assertIn("IP=", s2)

    def test_step_prompts_include_expected_labels(self) -> None:
        prefix, e1, e2 = split_prompt(self.prompt)
        s1, _ = split_target(self.target)
        self.assertIn("TASK: chained_step1", build_step1_prompt(prefix, e1))
        step2 = build_step2_prompt(prefix, s1, e2)
        self.assertIn("TASK: chained_step2", step2)
        self.assertIn("S1", step2)


if __name__ == "__main__":
    unittest.main()
