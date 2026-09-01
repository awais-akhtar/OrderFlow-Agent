from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orderflow_agent.agent import ConversationalTaskAgent
from orderflow_agent.knowledge import KnowledgeService
from orderflow_agent.runtime.rag import DualRAGEngine, make_chunks
from orderflow_agent.storage import SQLiteStorageAdapter


class StorageAdapterTest(unittest.TestCase):
    def test_round_trip_session_transcript_trace_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = SQLiteStorageAdapter(Path(directory) / "orderflow.db")
            agent = ConversationalTaskAgent(storage=storage)
            session = agent.open_session()
            agent.handle("add one medium cheese pizza", session)
            agent.handle("confirm order", session)
            confirmed = agent.handle("yes", session)

            turns = storage.list_turns(session.session_id)
            traces = storage.list_tool_traces(session.session_id)
            orders = storage.list_orders()
            csv_export = storage.export_orders_csv()
            json_export = storage.export_orders_json()

        self.assertIsNotNone(confirmed.confirmed_order_id)
        self.assertEqual(len(turns), 6)
        self.assertEqual(len(traces), 3)
        self.assertEqual(orders[0]["total"], 1399)
        self.assertEqual(orders[0]["fulfilment"], "pickup")
        self.assertEqual(orders[0]["delivery_address"], "")
        self.assertIn("Medium Cheese Pizza", csv_export)
        self.assertIn("fulfilment", csv_export)
        self.assertEqual(json.loads(json_export)[0]["total"], 1399)

    def test_persists_document_lineage_and_retrieval_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = SQLiteStorageAdapter(Path(directory) / "orderflow.db")
            text = "An existing-order complaint is transferred with the preserved cart and tool history."
            chunks = make_chunks("doc-1", "Handover", "handover.md", text, chunk_size=300, overlap=20)
            document_id, created = storage.save_knowledge_document(
                title="Handover",
                source_name="handover.md",
                mime_type="text/markdown",
                text=text,
                chunks=chunks,
            )
            result = DualRAGEngine(storage.list_knowledge_chunks()).retrieve("What is transferred?")
            storage.save_retrieval_trace(result, provider_id="retrieval-only", answer_generated=False)

            documents = storage.list_knowledge_documents()
            traces = storage.list_retrieval_traces()

        self.assertTrue(created)
        self.assertEqual(document_id, "doc-1")
        self.assertEqual(documents[0]["chunk_count"], 1)
        self.assertEqual(traces[0]["provider_id"], "retrieval-only")

    def test_builtin_knowledge_seed_refreshes_without_deleting_other_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = SQLiteStorageAdapter(root / "orderflow.db")
            source = root / "knowledge_base.md"
            source.write_text("# Rules\n\nThe first ordering rule requires explicit confirmation.", encoding="utf-8")
            service = KnowledgeService(storage)
            service.seed(source)
            service.ingest(
                "store-hours.md",
                "text/markdown",
                b"# Store hours\n\nThe demonstration store closes at ten in the evening.",
            )
            source.write_text("# Rules\n\nThe updated ordering rule requires a separate explicit yes.", encoding="utf-8")
            service.seed(source)
            documents = storage.list_knowledge_documents(include_text=True)

        self.assertEqual(len(documents), 2)
        seeded = next(row for row in documents if row["source_name"] == "knowledge_base.md")
        self.assertIn("separate explicit yes", seeded["text"])
        self.assertTrue(any(row["source_name"] == "store-hours.md" for row in documents))

    def test_handover_queue_and_live_messages_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = SQLiteStorageAdapter(Path(directory) / "orderflow.db")
            agent = ConversationalTaskAgent(storage=storage)
            first = agent.handle("I need a staff", agent.open_session())
            second = agent.handle("Please connect me to a staff", agent.open_session())

            self.assertEqual(storage.handover_queue_position(first.handover_case_id), 1)
            self.assertEqual(storage.handover_queue_position(second.handover_case_id), 2)
            customer_message = storage.append_handover_message(
                first.handover_case_id,
                role="customer",
                content="I can provide more detail.",
            )
            staff_message = storage.append_handover_message(
                first.handover_case_id,
                role="staff",
                content="I am reviewing this now.",
            )
            stored = storage.get_handover(first.handover_case_id)
            storage.complete_handover(
                first.handover_case_id,
                human_response=staff_message["content"],
            )

            self.assertEqual(customer_message["role"], "customer")
            self.assertEqual([row["role"] for row in stored["handover"]["live_messages"]], ["customer", "staff"])
            self.assertIsNone(storage.handover_queue_position(first.handover_case_id))
            self.assertEqual(storage.handover_queue_position(second.handover_case_id), 1)


if __name__ == "__main__":
    unittest.main()
