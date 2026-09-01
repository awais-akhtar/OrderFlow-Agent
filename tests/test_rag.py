import unittest

from orderflow_agent.runtime.rag import (
    DualRAGEngine,
    KnowledgeChunk,
    RAGError,
    build_grounded_prompt,
    extract_document_text,
    make_chunks,
)


class FakeEmbedder:
    provider_id = "fake"

    def embed_texts(self, texts):
        return [
            [
                float("refund" in text.lower() or "money" in text.lower()),
                float("handover" in text.lower() or "transfer" in text.lower()),
                float("weather" in text.lower()),
            ]
            for text in texts
        ]


class DualRAGTest(unittest.TestCase):
    def setUp(self):
        self.chunks = (
            KnowledgeChunk("c1", "d1", "Refund policy", "policy.md", "A refund returns money after approval.", 0, "a"),
            KnowledgeChunk("c2", "d2", "Handover", "handover.md", "A staff handover transfers the case summary.", 0, "b"),
            KnowledgeChunk("c3", "d3", "Weather", "weather.md", "Rain is expected tomorrow.", 0, "c"),
        )

    def test_fuses_lexical_and_provider_embedding_lanes(self):
        result = DualRAGEngine(self.chunks, embedder=FakeEmbedder()).retrieve("How is money returned?", top_k=2)

        self.assertEqual(result.hits[0].document_id, "d1")
        self.assertEqual(result.trace.mode, "dual-lane")
        self.assertEqual(result.trace.vector_lane, "fake-embeddings")
        self.assertGreater(result.trace.timings_ms["fusion_and_total"], 0)

    def test_local_vector_lane_is_explicit(self):
        result = DualRAGEngine(self.chunks).retrieve("transfer case", top_k=1)

        self.assertEqual(result.hits[0].document_id, "d2")
        self.assertIn(result.trace.vector_lane, {"local-lsa-vector", "local-term-vector"})

    def test_document_extraction_and_chunk_lineage(self):
        text = extract_document_text("notes.json", b'{"topic": "handover", "detail": "shared context"}')
        chunks = make_chunks("doc-1", "Notes", "notes.json", text, chunk_size=300, overlap=20)

        self.assertEqual(chunks[0].document_id, "doc-1")
        self.assertEqual(chunks[0].source_name, "notes.json")
        self.assertTrue(chunks[0].checksum)

    def test_rejects_unsupported_upload(self):
        with self.assertRaises(RAGError):
            extract_document_text("payload.exe", b"not a document")

    def test_grounded_prompt_labels_sources(self):
        result = DualRAGEngine(self.chunks).retrieve("refund", top_k=1)
        prompt = build_grounded_prompt("Can I get money back?", result.hits)
        self.assertIn("[S1]", prompt)
        self.assertIn("Answer only from the evidence", prompt)


if __name__ == "__main__":
    unittest.main()
