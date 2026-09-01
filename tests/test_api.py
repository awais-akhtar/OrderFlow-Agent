from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from orderflow_agent.gradio_ui import app


class ProductAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_order_and_handover_collections_are_available(self) -> None:
        for path, key in (("/api/orders", "orders"), ("/api/handovers", "handovers")):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("count", response.json())
                self.assertIn(key, response.json())

    def test_session_context_has_stable_empty_shape(self) -> None:
        response = self.client.get("/api/sessions/not-a-session/context")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["signals"], [])
        self.assertEqual(response.json()["turns"], [])
        self.assertEqual(response.json()["tool_traces"], [])


if __name__ == "__main__":
    unittest.main()
