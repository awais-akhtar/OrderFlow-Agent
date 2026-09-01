from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from orderflow_agent.gradio_ui import app, demo


class GradioSmokeTest(unittest.TestCase):
    def test_gradio_blocks_and_product_api_are_mounted(self) -> None:
        self.assertEqual(demo.title, "OrderFlow-Agent")
        paths = {getattr(route, "path", "") for route in app.routes}
        self.assertTrue({"/api/health", "/api/orders", "/api/handovers"} <= paths)

    def test_health_reports_model_configuration_without_secrets(self) -> None:
        response = TestClient(app).get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["application"], "OrderFlow-Agent")
        self.assertNotIn("api_key", payload)


if __name__ == "__main__":
    unittest.main()
