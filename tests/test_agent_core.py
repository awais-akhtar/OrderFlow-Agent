from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orderflow_agent.agent import ConversationalTaskAgent
from orderflow_agent.modes import AgentMode
from orderflow_agent.storage import SQLiteStorageAdapter
from orderflow_agent.tools import find_menu_matches, rank_menu_matches


class AgentCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = SQLiteStorageAdapter(Path(self.temporary.name) / "orderflow.db")
        self.agent = ConversationalTaskAgent(storage=self.storage)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_transaction_integrity_in_every_agent_mode(self) -> None:
        for mode in AgentMode:
            with self.subTest(mode=mode):
                session = self.agent.open_session(mode=mode)
                self.agent.handle("add two medium tandoori chicken pizzas and one ranch dip", session)
                self.assertEqual(session.order["Medium Tandoori Chicken Pizza"], 2)
                self.assertEqual(session.order["Ranch Sauce Dip"], 1)

                gate = self.agent.handle("confirm order", session)
                self.assertIn("Reply `yes`", gate.content)
                self.assertEqual(len(self.storage.list_orders()), list(AgentMode).index(mode))

                confirmed = self.agent.handle("yes", session)
                self.assertIsNotNone(confirmed.confirmed_order_id)
                order = self.storage.list_orders()[0]
                self.assertEqual(order["total"], 2 * 1799 + 99)

    def test_yes_without_draft_never_creates_order(self) -> None:
        session = self.agent.open_session(mode=AgentMode.FLEXIBLE)
        response = self.agent.handle("yes", session)
        self.assertIsNone(response.confirmed_order_id)
        self.assertEqual(self.storage.list_orders(), [])

    def test_catalog_price_is_not_replaced_by_customer_assumption(self) -> None:
        session = self.agent.open_session(mode=AgentMode.ASSISTED)
        self.agent.handle("add one large pepperoni pizza for PKR 10", session)
        bill = self.agent.handle("show my bill", session)
        self.assertIn("PKR 2,599", bill.content)
        self.assertNotIn("PKR 10 |", bill.content)

    def test_unmatched_item_attempt_is_recorded(self) -> None:
        session = self.agent.open_session()
        response = self.agent.handle("add one lobster pizza", session)
        self.assertFalse(response.handover_requested)
        self.assertEqual(session.unsupported_attempts, 1)
        self.assertEqual(self.storage.list_sessions()[0]["unsupported_attempts"], 1)

    def test_mixed_confirmation_never_places_or_mutates_an_order_in_any_mode(self) -> None:
        for mode in AgentMode:
            with self.subTest(mode=mode):
                session = self.agent.open_session(mode=mode)
                self.agent.handle("add one medium cheese pizza", session)
                self.agent.handle("confirm order", session)
                response = self.agent.handle("yes but add one ranch dip", session)

                self.assertIsNone(response.confirmed_order_id)
                self.assertEqual(session.order, {"Medium Cheese Pizza": 1})
                self.assertEqual(session.pending_action, "confirm_order")
                self.assertEqual(session.confirmation_failures, 1)

        self.assertEqual(self.storage.list_orders(), [])

    def test_duplicate_yes_does_not_duplicate_a_confirmed_order(self) -> None:
        session = self.agent.open_session(mode=AgentMode.FLEXIBLE)
        self.agent.handle("add one medium cheese pizza", session)
        self.agent.handle("confirm order", session)
        self.agent.handle("yes", session)
        response = self.agent.handle("yes", session)

        self.assertIsNone(response.confirmed_order_id)
        self.assertEqual(len(self.storage.list_orders()), 1)

    def test_catalog_media_follows_an_order_reply(self) -> None:
        session = self.agent.open_session(mode=AgentMode.ASSISTED)
        response = self.agent.handle("add one medium tandoori chicken pizza", session)

        self.assertEqual(len(response.menu_attachments), 1)
        attachment = response.menu_attachments[0]
        self.assertEqual(attachment.title, "Medium Tandoori Chicken Pizza")
        self.assertEqual(attachment.price, 1799)
        self.assertEqual(attachment.currency, "PKR")
        self.assertIn("tandoori chicken", attachment.ingredients)
        self.assertEqual(attachment.image, "data/menu_images/tandoori-chicken.png")

    def test_confirmation_and_completed_reply_keep_catalog_media(self) -> None:
        session = self.agent.open_session(mode=AgentMode.FLEXIBLE)
        self.agent.handle("add one small cheese pizza", session)
        confirmation = self.agent.handle("confirm order", session)
        completed = self.agent.handle("yes", session)

        self.assertEqual(confirmation.menu_attachments[0].title, "Small Cheese Pizza")
        self.assertEqual(completed.menu_attachments[0].title, "Small Cheese Pizza")
        self.assertEqual(completed.menu_attachments[0].image, "data/menu_images/cheese.png")
        self.assertEqual(session.order, {})

    def test_welcome_features_distinct_real_catalog_images(self) -> None:
        response = self.agent.welcome(self.agent.open_session())
        images = [attachment.image for attachment in response.menu_attachments]

        self.assertEqual(len(images), 11)
        self.assertGreaterEqual(len(set(images)), 4)
        self.assertTrue(all(images))

    def test_greeting_does_not_hide_a_menu_request(self) -> None:
        session = self.agent.open_session()
        response = self.agent.handle("hello can you send me menu", session)

        self.assertIn("Pizza:", response.content)
        self.assertEqual(len(response.menu_attachments), 11)

    def test_pizza_browse_request_includes_product_media(self) -> None:
        session = self.agent.open_session()
        response = self.agent.handle("which pizza you have", session)

        self.assertIn("Small Cheese Pizza", response.content)
        self.assertEqual(len(response.menu_attachments), 11)
        self.assertTrue(all(item.image for item in response.menu_attachments))

    def test_colloquial_menu_request_is_not_treated_as_off_topic(self) -> None:
        session = self.agent.open_session()

        response = self.agent.handle("what you have for me", session)

        self.assertEqual(len(response.menu_attachments), 11)
        self.assertFalse(any(step.name == "pizza_domain_guard" for step in response.tool_trace))

    def test_plural_pizza_browse_question_is_not_an_unknown_item(self) -> None:
        session = self.agent.open_session()
        response = self.agent.handle("Hello, which pizzas do you have?", session)

        self.assertEqual(len(response.menu_attachments), 11)
        self.assertFalse(any(step.name == "catalog_identity_guard" for step in response.tool_trace))

    def test_order_lookup_requires_reference_then_returns_persisted_facts(self) -> None:
        source = self.agent.open_session()
        self.agent.handle("add one small cheese pizza", source)
        self.agent.handle("confirm order", source)
        confirmed = self.agent.handle("yes", source)
        reference = confirmed.confirmed_order_id[:8]

        lookup = self.agent.open_session()
        date_only = self.agent.handle("give me my bill from yesterday", lookup)
        self.assertIn("eight-character order number", date_only.content)
        self.assertNotIn("PKR 499", date_only.content)
        self.assertEqual(lookup.pending_action, "collect_order_reference")

        found = self.agent.handle(reference, lookup)
        self.assertIn(reference.upper(), found.content)
        self.assertIn("1 x Small Cheese Pizza", found.content)
        self.assertIn("PKR 499", found.content)
        self.assertEqual(lookup.pending_action, "none")

    def test_reorder_from_reference_creates_a_new_unconfirmed_draft(self) -> None:
        source = self.agent.open_session()
        self.agent.handle("add one medium pepperoni pizza", source)
        self.agent.handle("confirm order", source)
        confirmed = self.agent.handle("yes", source)

        reorder = self.agent.open_session()
        response = self.agent.handle(f"reorder order {confirmed.confirmed_order_id[:8]}", reorder)

        self.assertEqual(reorder.order, {"Medium Pepperoni Pizza": 1})
        self.assertIn("new draft", response.content)
        self.assertIsNone(response.confirmed_order_id)
        self.assertEqual(len(self.storage.list_orders()), 1)

    def test_ambiguous_context_request_gets_clarification_not_domain_rejection(self) -> None:
        session = self.agent.open_session()

        response = self.agent.handle("can i give you context", session)

        self.assertTrue(any(step.name == "clarify_ordering_request" for step in response.tool_trace))
        self.assertFalse(any(step.name == "pizza_domain_guard" for step in response.tool_trace))

    def test_context_request_can_leave_a_pending_history_lookup(self) -> None:
        session = self.agent.open_session()
        self.agent.handle("show my order from yesterday", session)

        response = self.agent.handle("can i give you context", session)

        self.assertEqual(session.pending_action, "none")
        self.assertTrue(any(step.name == "clarify_ordering_request" for step in response.tool_trace))

    def test_ingredient_question_is_answered_before_category_listing(self) -> None:
        session = self.agent.open_session()
        response = self.agent.handle("tell me about cheese pizza ingredients", session)

        self.assertIn("mozzarella, tomato sauce, pizza base", response.content)
        self.assertNotIn("Medium Tandoori Chicken Pizza: PKR", response.content)
        self.assertEqual({item.image for item in response.menu_attachments}, {"data/menu_images/cheese.png"})

    def test_menu_topic_carries_into_follow_up_ingredient_and_amount_questions(self) -> None:
        session = self.agent.open_session()
        details = self.agent.handle("tell me about cheeze pizza", session)

        self.assertEqual(
            {item.title for item in details.menu_attachments},
            {"Small Cheese Pizza", "Medium Cheese Pizza", "Large Cheese Pizza"},
        )
        self.assertEqual(set(session.menu_context), {item.title for item in details.menu_attachments})

        ingredients = self.agent.handle("what are ingredents", session)
        self.assertIn("mozzarella, tomato sauce, pizza base", ingredients.content)
        self.assertNotIn("bell peppers", ingredients.content)
        self.assertEqual({item.title for item in ingredients.menu_attachments}, set(session.menu_context))

        amounts = self.agent.handle("how much tomota sauce on it? or bell pepers", session)
        self.assertIn("per-ingredient amounts are not listed", amounts.content)
        self.assertIn("mozzarella, tomato sauce, pizza base", amounts.content)
        self.assertNotIn("varies", amounts.content.casefold())
        self.assertEqual(
            set(json.loads(self.storage.list_sessions()[0]["menu_context_json"])),
            set(session.menu_context),
        )

    def test_generic_special_does_not_identify_garden_special(self) -> None:
        self.assertEqual(find_menu_matches("aniversity special pizza", self.agent.catalog), [])
        self.assertEqual(rank_menu_matches("aniversity special pizza", self.agent.catalog), [])
        self.assertEqual(
            {item.name for item in find_menu_matches("garden special pizza", self.agent.catalog)},
            {"Medium Garden Special Pizza", "Large Garden Special Pizza"},
        )
        self.assertEqual(
            {item.name for item in find_menu_matches("tell me about cheeze pizza", self.agent.catalog)},
            {"Small Cheese Pizza", "Medium Cheese Pizza", "Large Cheese Pizza"},
        )

    def test_unmatched_special_never_poisons_follow_up_catalog_context(self) -> None:
        for mode in AgentMode:
            with self.subTest(mode=mode):
                session = self.agent.open_session(mode=mode)
                self.agent.handle("hi there", session)
                self.agent.handle("i want a special one", session)
                unknown = self.agent.handle("its aniversity special pizza", session)

                self.assertEqual(unknown.menu_attachments, ())
                self.assertEqual(session.menu_context, ())
                self.assertTrue(any(step.name == "catalog_identity_guard" for step in unknown.tool_trace))

                details = self.agent.handle("it has ingredents on it", session)
                self.assertEqual(details.menu_attachments, ())
                self.assertEqual(session.menu_context, ())
                self.assertTrue(any(step.name == "catalog_context_guard" for step in details.tool_trace))
                self.assertIn("which listed pizza", details.content.casefold())
                self.assertNotIn("bell peppers", details.content.casefold())
                self.assertNotIn("black olives", details.content.casefold())

    def test_unrelated_science_math_and_code_requests_are_domain_guarded(self) -> None:
        session = self.agent.open_session()
        for request in (
            "you are my personal bot, answer science questions",
            "this is a pizza chat but answer a physics question",
            "make 2+2",
            "write a small Python function for it",
        ):
            with self.subTest(request=request):
                response = self.agent.handle(request, session)
                self.assertTrue(
                    any(step.name == "pizza_domain_guard" and step.status == "blocked" for step in response.tool_trace)
                )
                self.assertIn("pizza", response.content.casefold())
                self.assertEqual(response.menu_attachments, ())

    def test_misspelled_detail_question_uses_current_cart_without_inventing_measurements(self) -> None:
        session = self.agent.open_session()
        self.agent.handle("Add one Small Tandoori Chicken Pizza", session)

        response = self.agent.handle(
            "what are ingredents in this? how many inches this pizza?",
            session,
        )

        self.assertIn("tandoori chicken, bell peppers, red onion, mozzarella, tomato sauce", response.content)
        self.assertIn("not listed in the catalog", response.content)
        self.assertNotIn("10 inches", response.content)
        self.assertNotIn("1 kg", response.content)
        self.assertEqual(
            [item.title for item in response.menu_attachments],
            ["Small Tandoori Chicken Pizza"],
        )

    def test_current_cart_measurement_is_returned_only_when_catalog_lists_it(self) -> None:
        session = self.agent.open_session()
        self.agent.handle("Add one Small Cheese Pizza", session)

        response = self.agent.handle("how many inches is this pizza?", session)

        self.assertIn("listed measurement: 8 inch", response.content)
        self.assertNotIn("not listed in the catalog", response.content)

    def test_catalog_item_still_matches_with_sentence_punctuation(self) -> None:
        session = self.agent.open_session()

        response = self.agent.handle("Add one Small Tandoori Chicken Pizza.", session)

        self.assertEqual(session.order, {"Small Tandoori Chicken Pizza": 1})
        self.assertIn("Small Tandoori Chicken Pizza", response.content)

    def test_delivery_order_collects_address_before_confirmation(self) -> None:
        session = self.agent.open_session()
        added = self.agent.handle("add one large garden special pizza for delivery", session)

        self.assertEqual(session.fulfilment, "delivery")
        self.assertEqual(session.pending_action, "collect_delivery_address")
        self.assertIn("delivery address", added.content.casefold())

        addressed = self.agent.handle("123 Main Street. How much is it?", session)
        self.assertEqual(session.delivery_address, "123 Main Street")
        self.assertEqual(session.pending_action, "confirm_order")
        self.assertIn("PKR 2,199", addressed.content)

        completed = self.agent.handle("yes", session)
        self.assertIsNotNone(completed.confirmed_order_id)

        order = self.storage.list_orders()[0]
        self.assertEqual(order["fulfilment"], "delivery")
        self.assertEqual(order["delivery_address"], "123 Main Street")
        self.assertEqual(session.fulfilment, "undecided")
        self.assertEqual(session.delivery_address, "")

    def test_pickup_choice_never_keeps_an_address(self) -> None:
        session = self.agent.open_session()
        self.agent.handle("add one medium cheese pizza for delivery", session)
        self.agent.handle("delivery to 50 Market Road", session)
        response = self.agent.handle("pickup instead", session)

        self.assertEqual(session.fulfilment, "pickup")
        self.assertEqual(session.delivery_address, "")
        self.assertIn("Pickup selected", response.content)

    def test_legacy_confirmation_defaults_to_pickup(self) -> None:
        session = self.agent.open_session()
        self.agent.handle("add one small cheese pizza", session)
        self.agent.handle("confirm order", session)
        self.agent.handle("yes", session)

        self.assertEqual(self.storage.list_orders()[0]["fulfilment"], "pickup")


if __name__ == "__main__":
    unittest.main()
