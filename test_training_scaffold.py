import json
import unittest

from test_paths import repo_path
from training.eval_checkpoint import validate_eval_config
from training.train_lora import build_plan, validate_config


class TrainingScaffoldTests(unittest.TestCase):
    def test_lora_config_builds_plan(self) -> None:
        config = json.loads(
            repo_path("training", "configs", "lora_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        validate_config(config)
        plan = build_plan(config)
        self.assertEqual(plan["training_mode"], "lora")
        self.assertEqual(plan["effective_batch_hint"], 8)
        self.assertEqual(plan["model_name"], "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertFalse(plan["trust_remote_code"])
        self.assertEqual(plan["max_steps"], 12)

    def test_eval_config_is_valid(self) -> None:
        config = json.loads(
            repo_path("training", "configs", "eval_phase2.json").read_text(
                encoding="utf-8"
            )
        )
        validate_eval_config(config)
        self.assertIn("single_step", config["benchmark_stages"])
        self.assertIn("avg_field_accuracy", config["metrics"])
        self.assertIn("phase2_freeze_path", config)

    def test_next2_delta_configs_are_valid(self) -> None:
        lora_config = json.loads(
            repo_path("training", "configs", "lora_next2_delta.json").read_text(
                encoding="utf-8"
            )
        )
        eval_config = json.loads(
            repo_path(
                "training", "configs", "eval_phase2_next2_delta.json"
            ).read_text(encoding="utf-8")
        )
        validate_config(lora_config)
        validate_eval_config(eval_config)
        self.assertEqual(
            lora_config["guardrails"]["first_run_purpose"], "next2_delta"
        )
        self.assertEqual(
            eval_config["benchmark_stages"], ["single_step", "next_2_steps"]
        )

    def test_next2_factorized_configs_are_valid(self) -> None:
        lora_config = json.loads(
            repo_path(
                "training", "configs", "lora_next2_factorized.json"
            ).read_text(encoding="utf-8")
        )
        eval_config = json.loads(
            repo_path(
                "training", "configs", "eval_phase2_next2_factorized.json"
            ).read_text(encoding="utf-8")
        )
        validate_config(lora_config)
        validate_eval_config(eval_config)
        self.assertEqual(
            lora_config["guardrails"]["first_run_purpose"], "next2_factorized"
        )
        self.assertEqual(
            eval_config["benchmark_stages"], ["single_step", "next_2_steps"]
        )

    def test_next2_factorized_360m_configs_are_valid(self) -> None:
        lora_config = json.loads(
            repo_path(
                "training", "configs", "lora_next2_factorized_360m.json"
            ).read_text(encoding="utf-8")
        )
        eval_config = json.loads(
            repo_path(
                "training",
                "configs",
                "eval_phase2_next2_factorized_360m.json",
            ).read_text(encoding="utf-8")
        )
        validate_config(lora_config)
        validate_eval_config(eval_config)
        self.assertEqual(
            lora_config["model_name"], "HuggingFaceTB/SmolLM2-360M-Instruct"
        )
        self.assertEqual(
            lora_config["guardrails"]["first_run_purpose"],
            "next2_factorized_360m",
        )
        self.assertEqual(
            eval_config["benchmark_stages"], ["single_step", "next_2_steps"]
        )

    def test_next2_slots_configs_are_valid(self) -> None:
        lora_config = json.loads(
            repo_path("training", "configs", "lora_next2_slots.json").read_text(
                encoding="utf-8"
            )
        )
        eval_config = json.loads(
            repo_path(
                "training", "configs", "eval_phase2_next2_slots.json"
            ).read_text(encoding="utf-8")
        )
        validate_config(lora_config)
        validate_eval_config(eval_config)
        self.assertEqual(
            lora_config["guardrails"]["first_run_purpose"], "next2_slots"
        )
        self.assertEqual(eval_config["benchmark_stages"], ["next_2_steps_slots"])

    def test_next2_effects_configs_are_valid(self) -> None:
        lora_config = json.loads(
            repo_path(
                "training", "configs", "lora_next2_effects.json"
            ).read_text(encoding="utf-8")
        )
        eval_config = json.loads(
            repo_path(
                "training", "configs", "eval_phase2_next2_effects.json"
            ).read_text(encoding="utf-8")
        )
        validate_config(lora_config)
        validate_eval_config(eval_config)
        self.assertEqual(
            lora_config["guardrails"]["first_run_purpose"], "next2_effects"
        )
        self.assertEqual(eval_config["benchmark_stages"], ["next_2_effects"])


if __name__ == "__main__":
    unittest.main()
