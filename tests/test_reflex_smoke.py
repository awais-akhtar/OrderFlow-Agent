from __future__ import annotations

import unittest
from pathlib import Path

from orderflow_reflex.orderflow_reflex import app


ROOT = Path(__file__).resolve().parent.parent


class ReflexSmokeTest(unittest.TestCase):
    def test_customer_and_staff_routes_are_registered(self) -> None:
        routes = {page.route for page in app._unevaluated_pages.values()}

        self.assertIn("index", routes)
        self.assertIn("staff", routes)

    def test_api_url_is_not_pinned_to_a_stale_backend_port(self) -> None:
        config_source = (ROOT / "rxconfig.py").read_text(encoding="utf-8")

        self.assertNotIn("api_url=", config_source)

    def test_reflex_is_the_primary_interface_without_customer_diagnostics(self) -> None:
        pages = list(app._unevaluated_pages.values())
        customer = next(page for page in pages if page.route == "index")
        rendered = str(customer.component())

        self.assertIn("Live ordering agent", rendered)
        self.assertIn("Add to order", rendered)
        self.assertNotIn("KernelLoom Python package", rendered)
        self.assertNotIn('"session_id"', rendered)
        self.assertNotIn("tool_trace", rendered)

    def test_staff_interface_exposes_ticket_workflow(self) -> None:
        pages = list(app._unevaluated_pages.values())
        staff = next(page for page in pages if page.route == "staff")
        rendered = str(staff.component())

        self.assertIn("Tickets", rendered)
        self.assertIn("Resolve ticket", rendered)
        self.assertIn("Reply to customer", rendered)


if __name__ == "__main__":
    unittest.main()
