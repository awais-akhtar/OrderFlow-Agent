"""Validated JSON catalog and a small editing adapter for operators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Protocol


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


@dataclass(frozen=True)
class CatalogItem:
    sku: str
    name: str
    category: str
    price: int
    aliases: tuple[str, ...] = ()
    description: str = ""
    tags: tuple[str, ...] = ()
    ingredients: tuple[str, ...] = ()
    image: str = ""
    interaction_stats: dict[str, float] = field(default_factory=dict)
    active: bool = True
    title: str = ""


@dataclass(frozen=True)
class Catalog:
    name: str
    currency: str
    version: str
    items: tuple[CatalogItem, ...]

    @property
    def active_items(self) -> tuple[CatalogItem, ...]:
        return tuple(item for item in self.items if item.active)

    @property
    def by_name(self) -> dict[str, CatalogItem]:
        return {item.name: item for item in self.active_items}


class CatalogStore(Protocol):
    def load(self) -> Catalog: ...
    def save(self, catalog: Catalog) -> None: ...


class JsonCatalogStore:
    def __init__(self, path: str | Path = DEFAULT_CATALOG_PATH) -> None:
        self.path = Path(path)

    def load(self) -> Catalog:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        items = tuple(
            CatalogItem(
                sku=str(row["sku"]).strip(),
                name=str(row["name"]).strip(),
                category=str(row["category"]).strip(),
                price=int(row["price"]),
                aliases=tuple(str(value).strip() for value in row.get("aliases", []) if str(value).strip()),
                description=str(row.get("description", "")).strip(),
                tags=tuple(str(value).strip() for value in row.get("tags", []) if str(value).strip()),
                ingredients=tuple(
                    str(value).strip() for value in row.get("ingredients", []) if str(value).strip()
                ),
                image=str(row.get("image", "")).strip(),
                interaction_stats={
                    str(key): float(value) for key, value in row.get("interaction_stats", {}).items()
                },
                active=bool(row.get("active", True)),
                title=str(row.get("title", row["name"])).strip(),
            )
            for row in payload["items"]
        )
        catalog = Catalog(
            name=str(payload.get("name", "OrderFlow demo catalog")),
            currency=str(payload.get("currency", "PKR")),
            version=str(payload.get("version", "1")),
            items=items,
        )
        self._validate(catalog)
        return catalog

    def save(self, catalog: Catalog) -> None:
        self._validate(catalog)
        payload = {
            "name": catalog.name,
            "version": catalog.version,
            "currency": catalog.currency,
            "items": [
                {
                    **asdict(item),
                    "aliases": list(item.aliases),
                    "tags": list(item.tags),
                    "ingredients": list(item.ingredients),
                }
                for item in catalog.items
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def upsert(self, item: CatalogItem) -> Catalog:
        catalog = self.load()
        items = list(catalog.items)
        index = next((index for index, current in enumerate(items) if current.sku == item.sku), None)
        if index is None:
            items.append(item)
        else:
            items[index] = item
        updated = replace(catalog, items=tuple(items))
        self.save(updated)
        return updated

    @staticmethod
    def _validate(catalog: Catalog) -> None:
        skus = [item.sku for item in catalog.items]
        names = [item.name.casefold() for item in catalog.items]
        if not catalog.currency.strip():
            raise ValueError("Catalog currency is required.")
        if len(skus) != len(set(skus)):
            raise ValueError("Catalog SKUs must be unique.")
        if len(names) != len(set(names)):
            raise ValueError("Catalog item names must be unique.")
        for item in catalog.items:
            if not item.sku or not item.name or not item.category:
                raise ValueError("Every catalog item needs a SKU, name, and category.")
            if item.price < 0:
                raise ValueError(f"Price cannot be negative: {item.name}")
            if item.image:
                image_path = Path(item.image)
                if image_path.is_absolute() or ".." in image_path.parts:
                    raise ValueError(f"Image path must stay inside the project: {item.name}")
                if image_path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    raise ValueError(f"Unsupported image type for {item.name}.")
