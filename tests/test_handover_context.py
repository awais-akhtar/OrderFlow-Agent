from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orderflow_agent.agent import ConversationalTaskAgent
from orderflow_agent.context import ConversationContextAnalyzer
from orderflow_agent.context.models import ConversationSignal
from orderflow_agent.handover.policy import HandoverPolicy
from orderflow_agent.models import ConversationSession, TranscriptTurn
from orderflow_agent.storage import SQLiteStorageAdapter


class ConversationContextTest(unittest.TestCase):
    def test_explicit_observable_phrase_produces_transparent_signal(self) -> None:
        turns = [TranscriptTurn("session", "customer", "You misunderstood; I meant the vegetarian pizza.")]
        signal = ConversationContextAnalyzer().analyze(turns, repair_requests=1)
        self.assertEqual(signal.label, "confused")
        self.assertIn("correction request", signal.evidence)
        self.assertEqual(signal.source_turns, (turns[0].id,))
        self.assertEqual(signal.method, "deterministic-observable-events")

    def test_frustration_signal_alone_does_not_force_handover(self) -> None:
        signal = ConversationSignal("frustrated", 0.8, "Explicit phrase", ("turn-1",))
        decision = HandoverPolicy().decide("This is frustrating", ConversationSession(), signal)
        self.assertFalse(decision.should_handover)
        self.assertEqual(decision.trigger, "supporting_conversation_signal")


class HandoverIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = SQLiteStorageAdapter(Path(self.temporary.name) / "orderflow.db")
        self.agent = ConversationalTaskAgent(storage=self.storage)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_explicit_human_request_preserves_history_cart_and_tools(self) -> None:
        session = self.agent.open_session()
        self.agent.handle("add one medium garden special pizza", session)
        response = self.agent.handle("Please transfer me to a staff", session)

        self.assertTrue(response.handover_requested)
        self.assertEqual(response.handover_decision.trigger, "explicit_human_request")
        case = self.storage.list_handovers()[0]["handover"]
        self.assertEqual(case["cart"], {"Medium Garden Special Pizza": 1})
        self.assertGreaterEqual(len(case["conversation_history"]), 4)
        self.assertGreaterEqual(len(case["tool_history"]), 2)
        self.assertIn("Customer request:", case["summary"])
        self.assertIn("Current order/cart: 1 x Medium Garden Special Pizza", case["summary"])
        self.assertIn("Suggested next action:", case["summary"])

    def test_repeated_misunderstanding_triggers_at_configured_threshold(self) -> None:
        session = self.agent.open_session()
        first = self.agent.handle("No, that is not what I meant; add one medium cheese pizza", session)
        self.assertFalse(first.handover_requested)
        second = self.agent.handle("No, that is not what I meant; add one large cheese pizza", session)
        self.assertTrue(second.handover_requested)
        self.assertEqual(second.handover_decision.trigger, "repeated_misunderstanding")

    def test_ordinary_negative_word_does_not_create_false_handover(self) -> None:
        session = self.agent.open_session()
        response = self.agent.handle("No mushrooms please, show me the menu", session)
        self.assertFalse(response.handover_requested)
        self.assertEqual(self.storage.list_handovers(), [])

    def test_negated_manager_transfer_keeps_the_ordering_flow_active(self) -> None:
        session = self.agent.open_session()

        response = self.agent.handle(
            "Do not transfer me to a manager; add one Small Cheese Pizza",
            session,
        )

        self.assertFalse(response.handover_requested)
        self.assertEqual(session.order, {"Small Cheese Pizza": 1})
        self.assertEqual(self.storage.list_handovers(), [])

    def test_repeated_validation_failures_use_the_validation_trigger(self) -> None:
        session = self.agent.open_session()
        first = self.agent.handle("add a lobster pizza", session)
        self.assertFalse(first.handover_requested)
        second = self.agent.handle("add a sushi pizza", session)
        self.assertTrue(second.handover_requested)
        self.assertEqual(second.handover_decision.trigger, "repeated_validation_failures")

    def test_allergy_and_existing_order_complaint_are_escalated(self) -> None:
        for message, trigger in (
            ("I have a severe nut allergy; is this safe?", "allergy_or_food_safety"),
            ("My order is late and I need help", "existing_order_complaint"),
        ):
            with self.subTest(message=message):
                session = self.agent.open_session()
                response = self.agent.handle(message, session)
                self.assertTrue(response.handover_requested)
                self.assertEqual(response.handover_decision.trigger, trigger)

    def test_natural_human_refund_and_quality_phrases_are_escalated(self) -> None:
        for message, trigger in (
            ("I need a person to help me", "explicit_human_request"),
            ("i need staff", "explicit_human_request"),
            ("can i have any member of staff assistance please?", "explicit_human_request"),
            ("I want to speak with staff or employ", "explicit_human_request"),
            ("get me any employe", "explicit_human_request"),
            ("My card was charged twice", "refund_or_payment_dispute"),
            ("The pizza arrived burnt", "existing_order_complaint"),
        ):
            with self.subTest(message=message):
                session = self.agent.open_session()
                response = self.agent.handle(message, session)
                self.assertTrue(response.handover_requested)
                self.assertEqual(response.handover_decision.trigger, trigger)

    def test_pending_handover_locks_automated_actions_without_duplicate_cases(self) -> None:
        session = self.agent.open_session()
        self.agent.handle("add one medium garden special pizza", session)
        handover = self.agent.handle("I need a human now", session)
        locked = self.agent.handle("confirm order", session)

        self.assertTrue(session.handover_active)
        self.assertEqual(session.pending_action, "handover")
        self.assertEqual(locked.handover_case_id, handover.handover_case_id)
        self.assertIn("ordering actions will stay paused", locked.content)
        self.assertEqual(session.order, {"Medium Garden Special Pizza": 1})
        self.assertEqual(self.storage.list_orders(), [])
        self.assertEqual(len(self.storage.list_handovers()), 1)
        stored_session = self.storage.list_sessions()[0]
        self.assertEqual(stored_session["handover_active"], 1)
        self.assertEqual(stored_session["handover_case_id"], handover.handover_case_id)

    def test_missing_delivery_and_free_replacement_request_is_escalated(self) -> None:
        session = self.agent.open_session()

        response = self.agent.handle(
            "It was ordered on 31/8/2026 and I did not get it; give me a free pizza",
            session,
        )

        self.assertTrue(response.handover_requested)
        self.assertEqual(response.handover_decision.trigger, "existing_order_complaint")
        self.assertIn("restaurant staff", response.content)


if __name__ == "__main__":
    unittest.main()
