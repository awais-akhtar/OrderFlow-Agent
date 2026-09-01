from __future__ import annotations

import unittest

from orderflow_agent.models import AgentResponse, MenuAttachment
from orderflow_agent.ui import WorkspaceState, _menu_image_url, _stream_chunks, app


class NiceGUISmokeTest(unittest.TestCase):
    def test_product_page_and_api_routes_are_registered(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertTrue({"/", "/api/orders", "/api/handovers", "/api/sessions/{session_id}/context"} <= paths)

    def test_catalog_asset_path_is_served_without_exposing_arbitrary_files(self) -> None:
        self.assertEqual(
            _menu_image_url("data/menu_images/garden-special.png"),
            "/menu-images/garden-special.png",
        )
        self.assertIsNone(_menu_image_url("README.md"))

    def test_progressive_chunks_reconstruct_validated_markdown_exactly(self) -> None:
        content = "Added to the draft:\n- 1 x Medium Garden Special Pizza\n\nSay `confirm order`."
        chunks = _stream_chunks(content, words_per_chunk=3)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), content)

    def test_workspace_serializes_catalog_media_for_chat(self) -> None:
        state = WorkspaceState()
        response = AgentResponse(
            "Added.",
            menu_attachments=(
                MenuAttachment(
                    "pizza-garden-medium",
                    "Medium Garden Special Pizza",
                    "Peppers, onion, olives, mushrooms, and mozzarella.",
                    ("bell peppers", "red onion", "mozzarella"),
                    "data/menu_images/garden-special.png",
                    1799,
                    "PKR",
                ),
            ),
        )

        message = state.add_response(response, progressive=True)

        self.assertTrue(message["streaming"])
        self.assertEqual(message["content"], "")
        self.assertEqual(message["attachments"][0]["price"], 1799)
        self.assertEqual(message["attachments"][0]["ingredients"][0], "bell peppers")


if __name__ == "__main__":
    unittest.main()
