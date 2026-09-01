"""Real Chromium smoke test for the customer and staff Reflex routes."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .wait_for_reflex import wait_for_reflex


@unittest.skipUnless(os.getenv("ORDERFLOW_RUN_BROWSER_E2E") == "1", "browser E2E is enabled in CI")
class ReflexBrowserE2ETest(unittest.TestCase):
    def test_customer_and_staff_surfaces(self) -> None:
        from playwright.sync_api import sync_playwright

        root = Path(__file__).resolve().parents[2]
        frontend_port = _free_port()
        backend_port = _free_port()
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["ORDERFLOW_DB_PATH"] = str(Path(directory) / "e2e.db")
            environment["GENAI_PROVIDER"] = "disabled"
            environment["REFLEX_FRONTEND_PORT"] = str(frontend_port)
            environment["REFLEX_BACKEND_PORT"] = str(backend_port)
            environment["REFLEX_API_URL"] = f"http://127.0.0.1:{backend_port}"
            environment["REFLEX_DIR"] = str(root / ".reflex")
            process_options: dict[str, object] = {}
            if os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                process_options["start_new_session"] = True
            process = subprocess.Popen(
                [sys.executable, "app.py"],
                cwd=root,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **process_options,
            )
            try:
                frontend_url = f"http://127.0.0.1:{frontend_port}"
                backend_url = f"http://127.0.0.1:{backend_port}"
                wait_for_reflex(frontend_url, backend_url)
                with sync_playwright() as playwright:
                    executable = os.getenv("ORDERFLOW_CHROME_EXECUTABLE", "").strip()
                    browser = playwright.chromium.launch(executable_path=executable or None)
                    page = browser.new_page(viewport={"width": 1440, "height": 960})
                    browser_errors: list[str] = []
                    page.on("pageerror", lambda error: browser_errors.append(str(error)))
                    page.on(
                        "console",
                        lambda message: browser_errors.append(message.text) if message.type == "error" else None,
                    )
                    page.goto(frontend_url, wait_until="networkidle")
                    page.get_by_text("Live ordering agent", exact=True).wait_for()
                    page.get_by_placeholder("Type your pizza order...").wait_for()
                    body = page.locator("body").inner_text()
                    self.assertNotIn("KernelLoom Python package", body)
                    self.assertNotIn('"session_id"', body)
                    self.assertNotIn("tool_trace", body)
                    self.assertEqual(page.get_by_role("button", name="Human", exact=True).count(), 0)

                    page.get_by_label("Add to order").first.click()
                    page.locator(".cart-box").get_by_text("Small Cheese Pizza", exact=True).wait_for()
                    page.get_by_text("The ordering assistant is temporarily unavailable.", exact=False).wait_for()

                    composer = page.get_by_placeholder("Type your pizza order...")
                    composer.fill("i need staff")
                    composer.press("Enter")
                    page.get_by_text("Support ticket", exact=True).wait_for()
                    page.get_by_text("Queue 1", exact=True).wait_for()
                    staff_composer = page.get_by_placeholder("Message restaurant staff...")
                    staff_composer.fill("I need help changing this order.")
                    staff_composer.press("Enter")
                    page.get_by_text("I need help changing this order.", exact=True).first.wait_for()

                    page.goto(f"{frontend_url}/staff", wait_until="networkidle")
                    page.get_by_text("Restaurant workspace", exact=True).wait_for()
                    expected_navigation = (
                        "Overview",
                        "Orders",
                        "Handovers",
                        "Catalogue",
                        "Menu intelligence",
                        "Evaluation",
                        "Knowledge",
                        "Tool traces",
                        "Settings",
                    )
                    for label in expected_navigation:
                        self.assertEqual(page.get_by_role("button", name=label, exact=True).count(), 1)

                    page.get_by_role("button", name="Orders", exact=True).click()
                    page.get_by_text("Order history", exact=True).wait_for()
                    with page.expect_download() as csv_download:
                        page.get_by_role("button", name="CSV", exact=True).click()
                    csv_path = Path(directory) / "orders.csv"
                    csv_download.value.save_as(csv_path)
                    self.assertTrue(csv_path.read_text(encoding="utf-8").startswith("created_at,id,"))
                    with page.expect_download() as json_download:
                        page.get_by_role("button", name="JSON", exact=True).click()
                    json_path = Path(directory) / "orders.json"
                    json_download.value.save_as(json_path)
                    self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), [])
                    page.get_by_role("button", name="Handovers", exact=True).click()
                    page.get_by_text("Tickets", exact=True).wait_for()
                    page.get_by_text("I need help changing this order.", exact=True).first.wait_for()
                    inbox_box = page.locator(".ticket-inbox").bounding_box()
                    workspace_box = page.locator(".ticket-workspace").bounding_box()
                    self.assertIsNotNone(inbox_box)
                    self.assertIsNotNone(workspace_box)
                    assert inbox_box is not None and workspace_box is not None
                    self.assertGreater(workspace_box["x"], inbox_box["x"] + inbox_box["width"])
                    self.assertLess(abs(workspace_box["y"] - inbox_box["y"]), 4)
                    reply = page.get_by_placeholder("Reply to customer...")
                    reply.fill("I have the order and can help from here.")
                    reply.press("Enter")
                    page.locator(".ticket-chat-staff").get_by_text(
                        "I have the order and can help from here.", exact=True
                    ).wait_for()
                    page.get_by_role("button", name="Resolve ticket", exact=True).click()
                    page.get_by_text("Resolved", exact=True).first.wait_for()
                    page.get_by_role("button", name="Catalogue", exact=True).click()
                    page.get_by_text("Edit item", exact=True).wait_for()
                    page.get_by_role("button", name="Menu intelligence", exact=True).click()
                    menu_query = page.get_by_placeholder("Describe a pizza preference")
                    menu_query.fill("something spicy and vegetarian")
                    page.get_by_role("button", name="Rank menu", exact=True).click()
                    page.get_by_text("Medium Garden Heat Pizza", exact=True).wait_for()
                    page.get_by_role("button", name="Evaluation", exact=True).click()
                    page.get_by_role("button", name="Run scenarios", exact=True).click()
                    page.get_by_text("scenario and mode checks passed.", exact=False).wait_for()
                    self.assertEqual(page.get_by_text("Fail", exact=True).count(), 0)
                    page.get_by_role("button", name="Knowledge", exact=True).click()
                    knowledge_query = page.get_by_placeholder("Ask an operating-policy question")
                    knowledge_query.fill("What confirmation is required before placing an order?")
                    page.get_by_role("button", name="Search knowledge", exact=True).click()
                    page.get_by_text("separate explicit yes response", exact=False).wait_for()
                    page.get_by_role("button", name="Tool traces", exact=True).click()
                    page.get_by_text("Deterministic tool traces", exact=True).wait_for()
                    page.get_by_role("button", name="Settings", exact=True).click()
                    page.get_by_text("Model runtime", exact=True).wait_for()
                    page.get_by_role("button", name="Check runtime", exact=True).click()
                    page.get_by_text("Unavailable:", exact=False).wait_for()

                    page.goto(frontend_url, wait_until="networkidle")
                    page.locator(".message-staff").get_by_text(
                        "I have the order and can help from here.", exact=True
                    ).wait_for()
                    page.get_by_text("Resolved", exact=True).wait_for()
                    page.get_by_placeholder("Type your pizza order...").wait_for()

                    mobile = browser.new_page(viewport={"width": 390, "height": 844})
                    mobile.goto(frontend_url, wait_until="networkidle")
                    mobile.get_by_placeholder("Type your pizza order...").wait_for()
                    self.assertTrue(
                        mobile.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                    )
                    mobile.close()
                    self.assertEqual(browser_errors, [])
                    browser.close()
            finally:
                _stop_process_tree(process)


def _free_port() -> int:
    with socket.socket() as source:
        source.bind(("127.0.0.1", 0))
        return int(source.getsockname()[1])


def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=5)
