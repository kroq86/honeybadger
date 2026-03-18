import unittest

from reference_vm import ParseError, parse_program, run_program


class ReferenceVMTests(unittest.TestCase):
    def test_add_and_write(self) -> None:
        program = """
        READ R1, input[0]
        READ R2, input[1]
        ADD R3, R1, R2
        WRITE output[0], R3
        HALT
        """
        state, trace = run_program(program, inputs={0: 7, 1: 5}, trace=True)

        self.assertEqual(state.output[0], 12)
        self.assertEqual(state.halted, 1)
        self.assertEqual(state.error, "NONE")
        self.assertEqual(len(trace), 5)
        self.assertIn("EXEC ADD R3, R1, R2", trace[2].serialize())

    def test_cmp_and_jz_takes_branch(self) -> None:
        program = """
        READ R1, input[0]
        READ R2, input[1]
        CMP R1, R2
        JZ EQUAL
        CONST R3, 0
        WRITE output[0], R3
        HALT
        EQUAL:
        CONST R3, 1
        WRITE output[0], R3
        HALT
        """
        state, _ = run_program(program, inputs={0: 9, 1: 9})

        self.assertEqual(state.output[0], 1)
        self.assertEqual(state.z, 1)
        self.assertEqual(state.error, "NONE")

    def test_jnz_skips_when_zero_flag_is_set(self) -> None:
        program = """
        CONST R1, 4
        CONST R2, 4
        CMP R1, R2
        JNZ NOT_EQUAL
        CONST R3, 11
        WRITE output[0], R3
        HALT
        NOT_EQUAL:
        CONST R3, 99
        WRITE output[0], R3
        HALT
        """
        state, _ = run_program(program)

        self.assertEqual(state.output[0], 11)
        self.assertEqual(state.halted, 1)

    def test_store_then_load(self) -> None:
        program = """
        CONST R1, 42
        STORE [10], R1
        LOAD R2, [10]
        WRITE output[0], R2
        HALT
        """
        state, _ = run_program(program)

        self.assertEqual(state.memory[10], 42)
        self.assertEqual(state.registers[2], 42)
        self.assertEqual(state.output[0], 42)

    def test_add_sets_carry_and_wraps(self) -> None:
        program = """
        CONST R1, 32767
        CONST R2, 1
        ADD R3, R1, R2
        HALT
        """
        state, _ = run_program(program)

        self.assertEqual(state.registers[3], -32768)
        self.assertEqual(state.n, 1)
        self.assertEqual(state.z, 0)
        self.assertEqual(state.c, 0)

    def test_sub_sets_zero_and_borrow(self) -> None:
        program = """
        CONST R1, 0
        CONST R2, 1
        SUB R3, R1, R2
        HALT
        """
        state, _ = run_program(program)

        self.assertEqual(state.registers[3], -1)
        self.assertEqual(state.z, 0)
        self.assertEqual(state.n, 1)
        self.assertEqual(state.c, 1)

    def test_missing_input_sets_error(self) -> None:
        program = """
        READ R1, input[0]
        HALT
        """
        state, _ = run_program(program)

        self.assertEqual(state.error, "OOB_INPUT")
        self.assertEqual(state.halted, 0)

    def test_loop_hits_max_steps(self) -> None:
        program = """
        START:
        JMP START
        """
        state, _ = run_program(program, max_steps=3)

        self.assertEqual(state.error, "MAX_STEPS_EXCEEDED")

    def test_parse_rejects_unresolved_label(self) -> None:
        with self.assertRaises(ParseError):
            parse_program("JMP MISSING")

    def test_test_and_ordered_jumps_route_signs(self) -> None:
        program = """
        READ R1, input[0]
        TEST R1
        JL NEG
        JG POS
        CONST R2, 0
        WRITE output[0], R2
        HALT
        NEG:
        CONST R2, -1
        WRITE output[0], R2
        HALT
        POS:
        CONST R2, 1
        WRITE output[0], R2
        HALT
        """
        negative_state, _ = run_program(program, inputs={0: -5})
        zero_state, _ = run_program(program, inputs={0: 0})
        positive_state, _ = run_program(program, inputs={0: 8})

        self.assertEqual(negative_state.output[0], -1)
        self.assertEqual(zero_state.output[0], 0)
        self.assertEqual(positive_state.output[0], 1)

    def test_cmp_with_ordered_jumps_compares_values(self) -> None:
        program = """
        READ R1, input[0]
        READ R2, input[1]
        CMP R1, R2
        JL LESS
        JG GREATER
        CONST R3, 0
        WRITE output[0], R3
        HALT
        LESS:
        CONST R3, -1
        WRITE output[0], R3
        HALT
        GREATER:
        CONST R3, 1
        WRITE output[0], R3
        HALT
        """
        less_state, _ = run_program(program, inputs={0: 2, 1: 9})
        equal_state, _ = run_program(program, inputs={0: 7, 1: 7})
        greater_state, _ = run_program(program, inputs={0: 9, 1: 2})

        self.assertEqual(less_state.output[0], -1)
        self.assertEqual(equal_state.output[0], 0)
        self.assertEqual(greater_state.output[0], 1)


if __name__ == "__main__":
    unittest.main()
