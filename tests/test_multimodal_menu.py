from __future__ import annotations

import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from orderflow_agent.catalog import JsonCatalogStore
from orderflow_agent.multimodal import LightweightMenuEncoder, MenuIntelligence, MenuInterestModel
from orderflow_agent.multimodal.evaluation import evaluate_menu_retrieval


ROOT = Path(__file__).resolve().parent.parent


class MultimodalMenuTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = JsonCatalogStore().load()

    def test_text_only_fallback_returns_catalog_items(self) -> None:
        intelligence = MenuIntelligence(self.catalog)
        results = intelligence.recommend("a vegetarian pizza similar to a chicken pizza", limit=4)
        self.assertTrue(results)
        self.assertTrue(all("vegetarian" in row.item.tags for row in results))
        self.assertTrue(all(row.image_score == 0 for row in results))
        self.assertTrue(all("text features" in row.reason for row in results))

    def test_spicy_vegetarian_constraints_surface_catalogued_alternative(self) -> None:
        results = MenuIntelligence(self.catalog).recommend(
            "something like spicy chicken pizza but vegetarian",
            limit=3,
        )
        self.assertEqual(results[0].item.sku, "pizza-garden-heat-medium")

    def test_lexical_overlap_keeps_all_garden_variants_above_generic_cheese(self) -> None:
        results = MenuIntelligence(self.catalog).recommend(
            "vegetarian pizza with olives and mushrooms",
            limit=3,
        )

        self.assertEqual(
            {row.item.sku for row in results},
            {"pizza-garden-medium", "pizza-garden-large", "pizza-garden-heat-medium"},
        )

    def test_missing_image_has_explicit_zero_feature_fallback(self) -> None:
        representation = LightweightMenuEncoder().represent("medium pepperoni pizza")
        self.assertFalse(representation.has_image)
        self.assertEqual(representation.image.shape, (20,))
        self.assertTrue(np.allclose(representation.image, 0))

    def test_image_features_are_combined_when_available(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), (190, 45, 30)).save(buffer, format="PNG")
        representation = LightweightMenuEncoder().represent("spicy pizza", buffer.getvalue())
        self.assertTrue(representation.has_image)
        self.assertFalse(np.allclose(representation.image, 0))
        self.assertGreater(representation.joint.size, representation.text.size)

    def test_invalid_catalog_image_falls_back_to_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_image = root / "bad.png"
            bad_image.write_bytes(b"not an image")
            item = replace(self.catalog.active_items[0], image="bad.png")
            catalog = replace(self.catalog, items=(item,))
            results = MenuIntelligence(catalog, asset_root=root).recommend("cheese pizza")
        self.assertEqual(results[0].image_score, 0.0)
        self.assertIn("text features", results[0].reason)

    def test_interest_model_requires_and_reports_synthetic_labels(self) -> None:
        payload = json.loads((ROOT / "data" / "synthetic_menu_interactions.json").read_text(encoding="utf-8"))
        model = MenuInterestModel(MenuIntelligence(self.catalog))
        report = model.fit(payload["labels"], dataset_label=payload["dataset_label"])
        self.assertEqual(report.dataset_label, "synthetic")
        self.assertEqual(report.sample_count, 12)
        first = self.catalog.active_items[0]
        self.assertIsInstance(model.predict(first), float)

    def test_interest_model_refuses_unlabelled_claim(self) -> None:
        model = MenuInterestModel(MenuIntelligence(self.catalog))
        with self.assertRaises(ValueError):
            model.fit({item.sku: index for index, item in enumerate(self.catalog.active_items[:8])}, dataset_label="customers")

    def test_synthetic_retrieval_cases_report_recall_and_mrr(self) -> None:
        report = evaluate_menu_retrieval(top_k=3)
        self.assertEqual(report.dataset_label, "synthetic-demo")
        self.assertEqual(report.case_count, 5)
        self.assertGreaterEqual(report.recall_at_k, 0.8)
        self.assertGreaterEqual(report.mean_reciprocal_rank, 0.8)


if __name__ == "__main__":
    unittest.main()
