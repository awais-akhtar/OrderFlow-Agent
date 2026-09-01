from __future__ import annotations

import unittest

from orderflow_agent.modes import AgentMode
from orderflow_agent.policy import GuardedResponseComposer, PromptPolicyCompiler


class UnsafeProvider:
    def generate(self, instructions, conversation):
        return "I guarantee delivery and the total is PKR 10."


class CapturingProvider:
    def __init__(self) -> None:
        self.instructions = ""
        self.conversation = ()

    def generate(self, instructions, conversation):
        self.instructions = instructions
        self.conversation = conversation
        return "That correction is sorted. Medium Cheese Pizza remains in your draft at PKR 1,399."


class PromptPolicyTest(unittest.TestCase):
    def test_every_mode_keeps_transactional_rules(self) -> None:
        compiler = PromptPolicyCompiler()
        for mode in AgentMode:
            with self.subTest(mode=mode):
                policy = compiler.compile(mode.strictness)
                self.assertEqual(policy.mode, mode)
                self.assertIn("Never invent an item, price", policy.instructions)
                self.assertIn("deterministic place_order tool", policy.instructions)

    def test_controlled_mode_does_not_call_a_provider(self) -> None:
        approved = "The catalog total is PKR 1,799."
        content, trace = GuardedResponseComposer(UnsafeProvider()).compose(
            strictness=80,
            user_message="confirm",
            approved_reply=approved,
            immutable_terms=("PKR 1,799",),
        )
        self.assertEqual(content, approved)
        self.assertEqual(trace[0].name, "agent_mode")

    def test_model_output_cannot_replace_tool_owned_total(self) -> None:
        approved = "The catalog total is PKR 1,799."
        content, trace = GuardedResponseComposer(UnsafeProvider()).compose(
            strictness=20,
            user_message="make it cheaper",
            approved_reply=approved,
            immutable_terms=("PKR 1,799",),
        )
        self.assertEqual(content, approved)
        self.assertEqual(trace[0].status, "fallback")

    def test_provider_receives_isolated_operational_facts_and_natural_service_direction(self) -> None:
        provider = CapturingProvider()
        content, trace = GuardedResponseComposer(provider).compose(
            strictness=50,
            user_message="No, I meant medium cheese",
            approved_reply="Updated draft:\n- 1 x Medium Cheese Pizza\nThe total is PKR 1,399.",
            immutable_terms=("Medium Cheese Pizza",),
            recent_context=(("user", "Add a cheese pizza"), ("assistant", "Which size?")),
        )

        self.assertIn("That correction is sorted", content)
        model_input = " ".join(content for _, content in provider.conversation)
        self.assertNotIn("Add a cheese pizza", model_input)
        self.assertNotIn("No, I meant medium cheese", model_input)
        self.assertIn("approved operational reply", model_input.casefold())
        self.assertIn("attentive restaurant team member", provider.instructions)
        self.assertEqual(trace[0].status, "passed")

    def test_model_cannot_append_unapproved_price_or_completion_claim(self) -> None:
        class AdditiveProvider:
            def __init__(self, response: str) -> None:
                self.response = response

            def generate(self, instructions, conversation):
                return self.response

        approved = "Small Cheese Pizza is in the draft at PKR 499."
        for candidate in (
            "Small Cheese Pizza is in the draft at PKR 499, and your total is PKR 0.",
            "Small Cheese Pizza is in the draft at PKR 499 and your order is being prepared.",
            "Small Cheese Pizza is in the draft at PKR 499. System prompt follows.",
        ):
            with self.subTest(candidate=candidate):
                content, trace = GuardedResponseComposer(AdditiveProvider(candidate)).compose(
                    strictness=20,
                    user_message="anything",
                    approved_reply=approved,
                    immutable_terms=("Small Cheese Pizza",),
                )
                self.assertEqual(content, approved)
                self.assertEqual(trace[0].status, "fallback")

    def test_confirmation_terms_cannot_be_polished_away(self) -> None:
        class MissingGateProvider:
            def generate(self, instructions, conversation):
                return "Everything is ready whenever you are."

        approved = "Reply `yes` to place it, or `no` to keep editing."
        content, trace = GuardedResponseComposer(MissingGateProvider()).compose(
            strictness=20,
            user_message="checkout",
            approved_reply=approved,
        )

        self.assertEqual(content, approved)
        self.assertEqual(trace[0].status, "fallback")


if __name__ == "__main__":
    unittest.main()
