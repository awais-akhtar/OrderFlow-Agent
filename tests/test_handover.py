from __future__ import annotations

import unittest

from orderflow_agent.context.models import ConversationSignal
from orderflow_agent.handover.models import HandoverCase, HandoverDecision
from orderflow_agent.handover.summary import build_handover_summary


class HandoverRecordTest(unittest.TestCase):
    def test_summary_contains_each_operator_field(self) -> None:
        summary = build_handover_summary(
            customer_request="Please fix my existing order",
            issue="The delivered pizza was incorrect.",
            cart={"Medium Garden Special Pizza": 1},
            actions_attempted=("load_catalog", "validate_order"),
            outstanding_problem="Existing order needs store review.",
            relevant_customer_context="Two corrections were recorded.",
            suggested_next_action="Locate the order and review the complaint.",
        )

        for label in (
            "Customer request:",
            "Issue:",
            "Current order/cart:",
            "Actions already attempted:",
            "Outstanding problem:",
            "Relevant customer context:",
            "Suggested next action:",
        ):
            self.assertIn(label, summary)

    def test_context_carryover_only_uses_explicit_cart_facts(self) -> None:
        case = HandoverCase(
            session_id="session",
            decision=HandoverDecision(True, "Customer requested a staff.", 1.0, "explicit_human_request"),
            signal=ConversationSignal("neutral", 0.6, "No explicit signal.", ()),
            customer_request="Please transfer me",
            issue="Staff requested",
            cart={"Medium Cheese Pizza": 2, "Ranch Sauce Dip": 1},
            conversation_history=(),
            tool_history=(),
            actions_attempted=(),
            outstanding_problem="Operator response required",
            relevant_customer_context="No non-neutral signal was recorded.",
            suggested_next_action="Continue with the customer.",
            summary="Summary",
            status="completed",
            human_response="I can help with the order.",
            facts_carried_forward=("2 x Medium Cheese Pizza",),
        )

        self.assertEqual(case.context_carryover_score, 0.5)


if __name__ == "__main__":
    unittest.main()
