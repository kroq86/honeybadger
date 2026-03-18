import unittest

from baseline_trainer import (
    Example,
    build_few_shot_prompt,
    canonicalize,
    chained_delta_metrics,
    exact_match,
    filter_examples_by_max_steps,
    next_2_effect_field_metrics,
    next_2_slots_field_metrics,
    repair_prediction_for_stage,
    select_few_shots,
    stage_system_prompt,
    state_field_metrics,
)


class BaselineTrainerTests(unittest.TestCase):
    def test_canonicalize_trims_outer_noise(self) -> None:
        text = "\nIP=1\nR0=0\n"
        self.assertEqual(canonicalize(text), "IP=1\nR0=0")

    def test_exact_match_uses_canonicalized_text(self) -> None:
        self.assertTrue(exact_match("IP=1\n", "IP=1"))
        self.assertFalse(exact_match("IP=2", "IP=1"))

    def test_stage_system_prompt_is_stage_specific(self) -> None:
        self.assertIn("Next canonical machine state only.", stage_system_prompt("single_step"))
        self.assertIn("Emit S1 and S2 only. S1 is after E1. S2 is after E2.", stage_system_prompt("next_2_steps"))
        self.assertIn("Emit only these labels in this exact order", stage_system_prompt("next_2_steps_slots"))
        self.assertIn("Describe the effect of each step", stage_system_prompt("next_2_effects"))
        self.assertIn("Full state trace only.", stage_system_prompt("short_trace"))
        self.assertIn("Final canonical machine state only.", stage_system_prompt("terminal_state"))

    def test_build_few_shot_prompt_contains_examples_and_query(self) -> None:
        prompt = build_few_shot_prompt(
            "single_step",
            [Example("single_step", "prog_a", "branch", "TASK A", "RESULT A")],
            Example("single_step", "prog_b", "branch", "TASK B", "RESULT B"),
        )
        self.assertIn("EXAMPLE 1", prompt)
        self.assertIn("QUERY", prompt)
        self.assertTrue(prompt.strip().endswith("TARGET"))

    def test_build_few_shot_prompt_is_compact_for_next_2_steps(self) -> None:
        prompt = build_few_shot_prompt(
            "next_2_steps",
            [Example("next_2_steps", "prog_a", "loop", "TASK A", "RESULT A")],
            Example("next_2_steps", "prog_b", "loop", "TASK B", "RESULT B"),
        )
        self.assertIn("X\nTASK A\n=\nRESULT A", prompt)
        self.assertIn("Q\nTASK B\n=", prompt)

    def test_state_field_metrics_reports_partial_match(self) -> None:
        metrics = state_field_metrics(
            "IP=2\nR0=0 R1=2 R2=1 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE",
            "IP=2\nR0=0 R1=2 R2=9 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE",
        )
        self.assertGreater(metrics["field_accuracy"], 0.0)
        self.assertFalse(metrics["register_matches"]["R2"])

    def test_select_few_shots_prefers_same_category_and_opcode(self) -> None:
        train = [
            Example("single_step", "prog_a", "branch", "P1", "T1", instruction="READ R1, input[0]"),
            Example("single_step", "prog_b", "loop", "P2", "T2", instruction="ADD R1, R1, R2"),
            Example("single_step", "prog_c", "branch", "P3", "T3", instruction="READ R2, input[1]"),
        ]
        query = Example("single_step", "prog_x", "branch", "Q", "TQ", instruction="READ R3, input[2]")
        selected = select_few_shots("single_step", train, query, train_shots=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].category, "branch")
        self.assertEqual(selected[0].instruction.split()[0], "READ")

    def test_select_few_shots_prefers_same_selection_key_for_next_2(self) -> None:
        train = [
            Example("next_2_steps", "prog_a", "search", "P1", "T1", selection_key="READ|READ"),
            Example("next_2_steps", "prog_b", "search", "P2", "T2", selection_key="READ|CONST"),
            Example("next_2_steps", "prog_c", "loop", "P3", "T3", selection_key="CONST|CONST"),
        ]
        query = Example("next_2_steps", "prog_x", "search", "Q", "TQ", selection_key="READ|READ")
        selected = select_few_shots("next_2_steps", train, query, train_shots=1)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].selection_key, "READ|READ")

    def test_repair_prediction_for_state_adds_missing_terminal_lines(self) -> None:
        repaired = repair_prediction_for_stage(
            "single_step",
            "IP=2\nR0=0 R1=2 R2=9 R3=0 R4=0 R5=0 R6=0 R7=0\nZ=0 N=0 C=0\nOUT[0]=_",
        )
        self.assertIn("HALTED=0", repaired)
        self.assertIn("ERROR=NONE", repaired)

    def test_repair_prediction_for_next2_slots_canonicalizes_slot_order(self) -> None:
        repaired = repair_prediction_for_stage(
            "next_2_steps_slots",
            "S2_ERROR=NONE\nS1_IP=1\nS1_REG=R0=0 R1=9 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nS1_FLAGS=Z=0 N=0 C=0\nS1_OUT=_\nS1_HALTED=0\nS1_ERROR=NONE\nS2_IP=2\nS2_REG=R0=0 R1=9 R2=2 R3=0 R4=0 R5=0 R6=0 R7=0\nS2_FLAGS=Z=0 N=0 C=0\nS2_OUT=_\nS2_HALTED=0",
        )
        self.assertTrue(repaired.startswith("S1_IP=1"))
        self.assertIn("S2_ERROR=NONE", repaired)

    def test_next2_slots_field_metrics_detect_exact_match(self) -> None:
        target = (
            "S1_IP=1\nS1_REG=R0=0 R1=9 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\nS1_FLAGS=Z=0 N=0 C=0\nS1_OUT=_\nS1_HALTED=0\nS1_ERROR=NONE\n"
            "S2_IP=2\nS2_REG=R0=0 R1=9 R2=2 R3=0 R4=0 R5=0 R6=0 R7=0\nS2_FLAGS=Z=0 N=0 C=0\nS2_OUT=_\nS2_HALTED=0\nS2_ERROR=NONE"
        )
        metrics = next_2_slots_field_metrics(target, target)
        self.assertEqual(metrics["field_accuracy"], 1.0)

    def test_next2_effect_field_metrics_detect_exact_match(self) -> None:
        target = (
            "STEP1_OP=READ\nSTEP1_DST=R1\nSTEP1_SRC=input[4]\nSTEP1_VALUE=9\nSTEP1_IP=1\nSTEP1_FLAGS=Z=0 N=0 C=0\nSTEP1_OUT=_\nSTEP1_HALTED=0\nSTEP1_ERROR=NONE\n"
            "STEP2_OP=READ\nSTEP2_DST=R2\nSTEP2_SRC=input[0]\nSTEP2_VALUE=2\nSTEP2_IP=2\nSTEP2_FLAGS=Z=0 N=0 C=0\nSTEP2_OUT=_\nSTEP2_HALTED=0\nSTEP2_ERROR=NONE"
        )
        metrics = next_2_effect_field_metrics(target, target)
        self.assertEqual(metrics["field_accuracy"], 1.0)

    def test_repair_prediction_for_target_anchor_effect_reconstructs_prefix(self) -> None:
        repaired = repair_prediction_for_stage(
            "next_2_effects_target_anchor",
            "READ\nSTEP1_DST=R1\nSTEP1_SRC=input[0]\nSTEP1_VALUE=2\nSTEP1_IP=1\nSTEP1_FLAGS=Z=0 N=0 C=0\nSTEP1_OUT=_\nSTEP1_HALTED=0\nSTEP1_ERROR=NONE\nSTEP2_OP=CONST\nSTEP2_DST=R2\nSTEP2_SRC=0\nSTEP2_VALUE=0\nSTEP2_IP=2\nSTEP2_FLAGS=Z=0 N=0 C=0\nSTEP2_OUT=_\nSTEP2_HALTED=0\nSTEP2_ERROR=NONE",
        )
        self.assertTrue(repaired.startswith("STEP1_OP=READ"))

    def test_filter_examples_by_max_steps_keeps_shorter_traces(self) -> None:
        examples = [
            Example("short_trace", "prog_a", "loop", "P1", "T1", num_steps=8),
            Example("short_trace", "prog_b", "loop", "P2", "T2", num_steps=12),
            Example("short_trace", "prog_c", "loop", "P3", "T3", num_steps=None),
        ]
        filtered = filter_examples_by_max_steps(examples, max_steps=10)
        self.assertEqual([example.program_name for example in filtered], ["prog_a", "prog_c"])

    def test_chained_delta_metrics_give_zero_credit_to_unchanged_fields(self) -> None:
        prompt = (
            "TASK: chained_step1\nINPUT\ninput[0]=9\nS0\n"
            "IP=0\nR0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\n"
            "Z=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE\nE1\nREAD R1, input[0]"
        )
        target = (
            "IP=1\nR0=0 R1=9 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\n"
            "Z=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
        )
        zero_like = (
            "IP=0\nR0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0\n"
            "Z=0 N=0 C=0\nOUT[0]=_\nHALTED=0\nERROR=NONE"
        )
        metrics = chained_delta_metrics(zero_like, target, prompt, "next_2_chained_step1")
        self.assertEqual(metrics["delta_transition_score"], 0.0)
        self.assertEqual(metrics["delta_exact"], 0.0)


if __name__ == "__main__":
    unittest.main()
