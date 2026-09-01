"""Similarity and constrained recommendation over the pizza catalog."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import re

from PIL import Image, UnidentifiedImageError

from orderflow_agent.catalog import Catalog, CatalogItem

from .backends import LightweightMenuEncoder, MenuEmbeddingBackend, cosine_similarity


REASON_STOPWORDS = {
    "a", "an", "and", "but", "for", "i", "in", "is", "it", "of", "or", "similar", "something",
    "the", "this", "to", "want", "with",
}


@dataclass(frozen=True)
class MenuRecommendation:
    item: CatalogItem
    score: float
    text_score: float
    image_score: float
    reason: str


class MenuIntelligence:
    def __init__(
        self,
        catalog: Catalog,
        *,
        backend: MenuEmbeddingBackend | None = None,
        asset_root: str | Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.backend = backend or LightweightMenuEncoder()
        self.asset_root = Path(asset_root) if asset_root else Path.cwd()

    def recommend(
        self,
        query: str,
        *,
        query_image: bytes | None = None,
        limit: int = 5,
    ) -> list[MenuRecommendation]:
        query_representation = self.backend.represent(query, query_image)
        candidates = self._apply_constraints(query, list(self.catalog.active_items))
        recommendations: list[MenuRecommendation] = []
        for item in candidates:
            item_image = self._read_image(item)
            item_representation = self.backend.represent(self._item_text(item), item_image)
            embedding_score = cosine_similarity(query_representation.text, item_representation.text)
            lexical_score = self._lexical_coverage(query, item)
            text_score = 0.7 * embedding_score + 0.3 * lexical_score
            image_score = (
                cosine_similarity(query_representation.image, item_representation.image)
                if query_representation.has_image and item_representation.has_image
                else 0.0
            )
            score = 0.78 * text_score + 0.22 * image_score
            recommendations.append(
                MenuRecommendation(
                    item=item,
                    score=round(score, 5),
                    text_score=round(text_score, 5),
                    image_score=round(image_score, 5),
                    reason=self._reason(
                        query,
                        item,
                        query_representation.has_image and item_representation.has_image,
                    ),
                )
            )
        recommendations.sort(key=lambda row: (-row.score, row.item.price, row.item.name))
        return recommendations[: max(1, min(limit, 20))]

    def representation(self, item: CatalogItem):
        return self.backend.represent(self._item_text(item), self._read_image(item))

    def _read_image(self, item: CatalogItem) -> bytes | None:
        if not item.image:
            return None
        path = (self.asset_root / item.image).resolve()
        root = self.asset_root.resolve()
        if root not in path.parents and path != root:
            return None
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            return data
        except (OSError, UnidentifiedImageError, ValueError):
            return None

    @staticmethod
    def _item_text(item: CatalogItem) -> str:
        return " ".join(
            value
            for value in (
                item.name,
                item.title,
                item.description,
                item.category,
                " ".join(item.ingredients),
                " ".join(item.tags),
                " ".join(item.aliases),
            )
            if value
        )

    @staticmethod
    def _apply_constraints(query: str, items: list[CatalogItem]) -> list[CatalogItem]:
        text = query.casefold()
        if "vegetarian" in text or "veggie" in text:
            items = [item for item in items if "vegetarian" in item.tags]
        if "pizza" in text:
            items = [item for item in items if item.category.casefold() == "pizza"]
        if "spicy" in text:
            spicy = [item for item in items if "spicy" in item.tags]
            if spicy:
                items = spicy
        return items

    @staticmethod
    def _lexical_coverage(query: str, item: CatalogItem) -> float:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.casefold())) - REASON_STOPWORDS
        if not query_tokens:
            return 0.0
        item_tokens = set(re.findall(r"[a-z0-9]+", MenuIntelligence._item_text(item).casefold()))
        return len(query_tokens.intersection(item_tokens)) / len(query_tokens)

    @staticmethod
    def _reason(query: str, item: CatalogItem, has_image: bool) -> str:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.casefold())) - REASON_STOPWORDS
        item_tokens = set(re.findall(r"[a-z0-9]+", MenuIntelligence._item_text(item).casefold()))
        matched = sorted(query_tokens.intersection(item_tokens))
        basis = "text and image features" if has_image else "text features"
        return f"Ranked from {basis}" + (f"; shared terms: {', '.join(matched[:4])}" if matched else "") + "."
