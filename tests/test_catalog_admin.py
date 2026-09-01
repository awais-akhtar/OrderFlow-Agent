from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orderflow_agent.catalog import CatalogItem, JsonCatalogStore


class CatalogAdministrationTest(unittest.TestCase):
    def test_rich_item_round_trips_through_json_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                json.dumps({"name": "Test menu", "version": "1", "currency": "PKR", "items": []}),
                encoding="utf-8",
            )
            store = JsonCatalogStore(path)
            item = CatalogItem(
                sku="pizza-test-medium",
                name="Medium Test Pizza",
                title="Test Pizza",
                category="Pizza",
                price=1200,
                aliases=("test pizza",),
                description="A catalog administration fixture.",
                tags=("vegetarian",),
                ingredients=("mozzarella", "tomato sauce"),
                image="data/menu_images/garden-special.png",
                interaction_stats={"synthetic_views": 12.0},
            )
            store.upsert(item)
            restored = store.load().items[0]

        self.assertEqual(restored, item)

    def test_negative_price_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                json.dumps({"name": "Test menu", "version": "1", "currency": "PKR", "items": []}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                JsonCatalogStore(path).upsert(CatalogItem("bad", "Bad Price", "Pizza", -1))

    def test_image_path_cannot_escape_the_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                json.dumps({"name": "Test menu", "version": "1", "currency": "PKR", "items": []}),
                encoding="utf-8",
            )
            item = CatalogItem("bad-image", "Bad Image", "Pizza", 100, image="../secret.png")
            with self.assertRaises(ValueError):
                JsonCatalogStore(path).upsert(item)


if __name__ == "__main__":
    unittest.main()
