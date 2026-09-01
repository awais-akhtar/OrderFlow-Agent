"""Deterministic menu matching, order mutation, validation, and billing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from .catalog import Catalog, CatalogItem
from .models import Bill, BillLine

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "couple": 2,
}
MAX_ITEM_QUANTITY = 20
MAX_ORDER_QUANTITY = 50
SIZE_WORDS = {"small", "medium", "large"}
MENU_FILLERS = {
    "about", "are", "available", "can", "do", "have", "how", "i", "in", "ingredient", "ingredients",
    "ingredent", "ingredents", "is", "it", "list", "made", "me", "menu", "much", "of", "on", "pizza",
    "please", "send", "show", "that", "tell", "the", "these", "this", "those", "what", "which", "with",
    "you", "your",
}
MENU_GENERIC_TERMS = MENU_FILLERS | SIZE_WORDS | {
    "classic", "flavor", "flavour", "get", "kind", "need", "one", "or", "order", "sauce", "special",
    "type", "want",
}
MENU_SUPPORTING_TERMS = {"classic", "special"}
MEASUREMENT_WORDS = {
    "centimeter", "centimeters", "centimetre", "centimetres", "cm", "diameter", "gram", "grams",
    "inch", "inches", "kg", "kilogram", "kilograms", "measurement", "measurements", "portion",
    "portions", "weight", "weights",
}
INGREDIENT_AMOUNT_WORDS = {
    "amount", "amounts", "gram", "grams", "kg", "kilogram", "kilograms", "portion", "portions",
    "quantity", "quantities", "weight", "weights",
}
INGREDIENT_HINTS = {
    "base", "cheese", "chicken", "ingredient", "ingredients", "mozzarella", "mushroom", "mushrooms",
    "olive", "olives", "onion", "onions", "pepper", "peppers", "peper", "pepers", "pepperoni", "sauce",
    "tomato", "tomota", "topping", "toppings",
}
CATALOG_MEASUREMENT_PATTERN = (
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*[- ]?"
    r"(?:inch(?:es)?|centimet(?:er|re)s?|cm|kg|kilograms?|grams?)\b"
)


def normalize_text(value: str) -> str:
    text = value.casefold().replace("&", " and ")
    text = re.sub(r"(\d)\s*(liters|liter|litres|litre|ltr)", r"\1ltr", text)
    text = re.sub(r"(\d)\s*ml", r"\1ml", text)
    text = text.replace("pizzas", "pizza").replace("dips", "dip").replace("sauces", "sauce")
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def asks_about_ingredients(text: str) -> bool:
    """Recognize common ingredient wording and close customer misspellings."""

    normalized = normalize_text(text)
    tokens = set(normalized.split())
    if (
        "what is in" in normalized
        or "what comes on" in normalized
        or "made with" in normalized
    ):
        return True
    return any(
        token in {"ingredient", "ingredients", "ingredent", "ingredents"}
        or SequenceMatcher(None, token, "ingredients").ratio() >= 0.84
        for token in tokens
    )


def asks_about_item_measurements(text: str) -> bool:
    return bool(set(normalize_text(text).split()).intersection(MEASUREMENT_WORDS))


def asks_about_ingredient_amount(text: str) -> bool:
    """Identify requests for ingredient quantities without treating a pizza price as an amount."""

    normalized = normalize_text(text)
    tokens = set(normalized.split())
    has_amount_cue = "how much" in normalized or bool(tokens.intersection(INGREDIENT_AMOUNT_WORDS))
    return has_amount_cue and bool(tokens.intersection(INGREDIENT_HINTS))


def asks_about_item_details(text: str) -> bool:
    return (
        asks_about_ingredients(text)
        or asks_about_item_measurements(text)
        or asks_about_ingredient_amount(text)
    )


def listed_measurements(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(CATALOG_MEASUREMENT_PATTERN, text, re.IGNORECASE)))


def _terms(item: CatalogItem) -> tuple[str, ...]:
    return tuple(sorted({normalize_text(value) for value in (item.name, *item.aliases)}, key=len, reverse=True))


def _menu_terms(item: CatalogItem) -> tuple[str, ...]:
    return tuple(
        sorted(
            {normalize_text(value) for value in (item.name, item.title, *item.aliases) if value},
            key=len,
            reverse=True,
        )
    )


@dataclass(frozen=True)
class MenuMatchEvidence:
    """A catalogue match with inspectable evidence rather than sentence-level similarity."""

    item: CatalogItem
    score: float
    reason: str
    matched_terms: tuple[str, ...]


def _quantity_before(text: str, start: int, lower_bound: int = 0) -> int:
    for token in reversed(text[lower_bound:start].split()[-5:]):
        if token.isdigit():
            if len(token) > 4:
                return MAX_ITEM_QUANTITY + 1
            try:
                return max(1, int(token))
            except ValueError:
                return MAX_ITEM_QUANTITY + 1
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    return 1


def invalid_order_quantity_reason(text: str, catalog: Catalog) -> str | None:
    """Reject quantity forms the deterministic parser cannot represent safely."""

    if not extract_order_from_text(text, catalog):
        return None
    lowered = unicodedata.normalize("NFKC", text.casefold()).translate(
        str.maketrans({"−": "-", "–": "-", "—": "-", "⁄": "/"})
    )
    unsupported_unicode_number = any(
        character.isnumeric() and not ("0" <= character <= "9")
        for character in text
    )
    unsupported_quantity = bool(
        unsupported_unicode_number
        or re.search(r"(?<![\w.])-\s*\d+(?:\.\d+)?\b", lowered)
        or re.search(r"\b(?:minus|negative)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", lowered)
        or re.search(r"\b0+\b", lowered)
        or re.search(r"\b\d+\.\d+\b", lowered)
        or re.search(r"(?<!\d)\.\d+\b|\b\d+\.{2,}\d*\b", lowered)
        or re.search(r"\b\d+\s*[,/:;|^]\s*\d+\b", lowered)
        or re.search(r"\b\d+\s*[-+]\s*\d+\b", lowered)
        or re.search(r"\b(?:\d+(?:\.\d+)?e[+-]?\d+|0x[0-9a-f]+)\b", lowered)
        or re.search(
            r"\b(?:half|quarter|nan|inf|infinite|infinity|few|several|double|triple|twice|thrice)\b|∞",
            lowered,
        )
        or re.search(r"\b(?:thirty|forty|fifty|sixty|seventy|eighty|ninety)\b", lowered)
        or re.search(
            r"\btwenty[-\s]+(?:one|two|three|four|five|six|seven|eight|nine)\b",
            lowered,
        )
        or re.search(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
            r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+(?:or|to)\s+"
            r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
            r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
            lowered,
        )
        or re.search(
            r"\b(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(?:dozen|hundred|thousand|million|billion)\b",
            lowered,
        )
        or re.search(r"\b(?:hundred|thousand|million|billion|trillion)\b", lowered)
        or re.search(r"\b\d+\s+\d+\b", lowered)
        or re.search(
            r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
            r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+"
            r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
            r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
            lowered,
        )
        or re.search(
            r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
            r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+\d+\b(?!\.)",
            lowered,
        )
        or re.search(r"\b(?:all|every)\b", lowered)
    )
    if unsupported_quantity:
        return f"A single menu item is limited to whole quantities from 1 to {MAX_ITEM_QUANTITY}."
    return None


def extract_order_from_text(text: str, catalog: Catalog) -> dict[str, int]:
    normalized = normalize_text(text)
    spans: list[tuple[int, int, str]] = []
    for item in catalog.active_items:
        for term in _terms(item):
            for match in re.finditer(rf"(?<![a-z0-9.]){re.escape(term)}(?![a-z0-9.])", normalized):
                spans.append((match.start(), match.end(), item.name))
    spans.sort(key=lambda row: (row[0], -(row[1] - row[0])))
    selected: list[tuple[int, int, str]] = []
    for span in spans:
        if any(span[0] < current[1] and span[1] > current[0] for current in selected):
            continue
        selected.append(span)
    order: dict[str, int] = {}
    previous_end = 0
    for start, end, name in selected:
        order[name] = order.get(name, 0) + _quantity_before(normalized, start, previous_end)
        previous_end = end
    return order


def missing_order_details_hint(text: str) -> str | None:
    normalized = normalize_text(text)
    if "pizza" in normalized and not SIZE_WORDS.intersection(normalized.split()):
        return "Which size would you like: small, medium, or large?"
    return None


def add_items(order: dict[str, int], items: dict[str, int], catalog: Catalog) -> dict[str, int]:
    available = catalog.by_name
    updated = dict(order)
    for name, quantity in items.items():
        if name in available:
            updated[name] = updated.get(name, 0) + max(1, quantity)
    return updated


def remove_items(order: dict[str, int], items: dict[str, int]) -> dict[str, int]:
    updated = dict(order)
    for name, quantity in items.items():
        if name in updated:
            updated[name] -= max(1, quantity)
            if updated[name] <= 0:
                del updated[name]
    return updated


def validate_order(order: dict[str, int], catalog: Catalog) -> tuple[bool, str]:
    if not order:
        return False, "The draft order is empty."
    if any(name not in catalog.by_name or quantity < 1 for name, quantity in order.items()):
        return False, "The draft contains an unavailable or invalid item."
    if any(quantity > MAX_ITEM_QUANTITY for quantity in order.values()):
        return False, f"A single menu item is limited to {MAX_ITEM_QUANTITY} per order."
    if sum(order.values()) > MAX_ORDER_QUANTITY:
        return False, f"An order is limited to {MAX_ORDER_QUANTITY} total items."
    return True, "The draft passed catalog and quantity validation."


def generate_bill(order: dict[str, int], catalog: Catalog) -> Bill:
    lines = tuple(
        BillLine(item.sku, item.name, quantity, item.price, item.price * quantity)
        for name, quantity in order.items()
        if (item := catalog.by_name.get(name)) is not None
    )
    return Bill(lines=lines, grand_total=sum(line.total for line in lines), currency=catalog.currency)


def format_money(amount: int, currency: str) -> str:
    return f"{currency} {amount:,}"


def format_order(order: dict[str, int]) -> str:
    return "\n".join(f"- {quantity} x {name}" for name, quantity in order.items()) or "No items yet."


def format_bill(order: dict[str, int], catalog: Catalog) -> str:
    bill = generate_bill(order, catalog)
    if not bill.lines:
        return "No bill yet."
    rows = ["| Item | Qty | Unit | Total |", "| --- | ---: | ---: | ---: |"]
    rows.extend(
        f"| {line.item} | {line.quantity} | {format_money(line.unit_price, bill.currency)} | "
        f"{format_money(line.total, bill.currency)} |"
        for line in bill.lines
    )
    rows.append(f"| **Grand total** |  |  | **{format_money(bill.grand_total, bill.currency)}** |")
    return "\n".join(rows)


def specific_menu_query_terms(query: str) -> set[str]:
    """Return customer terms capable of identifying an item or family by themselves."""

    return set(normalize_text(query).split()) - MENU_GENERIC_TERMS


def rank_menu_matches(query: str, catalog: Catalog, limit: int = 8) -> list[MenuMatchEvidence]:
    """Rank menu items only when the query contains positive catalogue identity evidence."""

    normalized = normalize_text(query)
    if not normalized:
        return []
    query_tokens = set(normalized.split())
    query_terms = query_tokens - MENU_GENERIC_TERMS
    if not query_terms:
        return []
    requested_sizes = query_tokens.intersection(SIZE_WORDS)
    requested_category = "pizza" if "pizza" in query_tokens else ""
    padded_query = f" {normalized} "
    ranked: list[MenuMatchEvidence] = []
    for item in catalog.active_items:
        item_tokens = set(
            normalize_text(" ".join((item.name, item.title, *item.aliases, *item.tags))).split()
        )
        if requested_category and normalize_text(item.category) != requested_category:
            continue
        if requested_sizes and not requested_sizes.intersection(item_tokens):
            continue
        item_identity = item_tokens - MENU_GENERIC_TERMS
        phrase_matches = tuple(
            term
            for term in _menu_terms(item)
            if (set(term.split()) - MENU_GENERIC_TERMS) and f" {term} " in padded_query
        )
        if phrase_matches:
            supporting = query_tokens.intersection(item_tokens).intersection(MENU_SUPPORTING_TERMS)
            ranked.append(
                MenuMatchEvidence(item, 1.0 + 0.02 * len(supporting), "exact_catalog_phrase", phrase_matches)
            )
            continue
        overlap = query_terms.intersection(item_identity)
        supporting = query_tokens.intersection(item_tokens).intersection(MENU_SUPPORTING_TERMS)
        if overlap:
            coverage = len(overlap) / max(1, len(query_terms))
            score = 0.86 + 0.1 * coverage + 0.02 * len(supporting)
            ranked.append(
                MenuMatchEvidence(item, score, "distinctive_token_overlap", tuple(sorted(overlap | supporting)))
            )
            continue
        similarities = [
            (SequenceMatcher(None, customer_term, catalog_term).ratio(), customer_term, catalog_term)
            for customer_term in query_terms
            for catalog_term in item_identity
            if min(len(customer_term), len(catalog_term)) >= 5
        ]
        if not similarities:
            continue
        similarity, customer_term, catalog_term = max(similarities)
        if similarity >= 0.8:
            ranked.append(
                MenuMatchEvidence(
                    item,
                    0.72 + 0.18 * similarity + 0.02 * len(supporting),
                    "bounded_token_typo",
                    (f"{customer_term}->{catalog_term}", *sorted(supporting)),
                )
            )
    catalog_order = {item.sku: index for index, item in enumerate(catalog.active_items)}
    if any(match.reason == "exact_catalog_phrase" for match in ranked):
        ranked = [match for match in ranked if match.reason == "exact_catalog_phrase"]
    ranked.sort(key=lambda match: (-match.score, catalog_order[match.item.sku]))
    return ranked[:limit]


def find_menu_matches(query: str, catalog: Catalog, limit: int = 8) -> list[CatalogItem]:
    return [match.item for match in rank_menu_matches(query, catalog, limit)]


def answer_menu_question(
    query: str,
    catalog: Catalog,
    context_items: Iterable[CatalogItem] = (),
) -> str:
    normalized = normalize_text(query)
    categories = tuple(dict.fromkeys(item.category for item in catalog.active_items))
    ingredient_request = asks_about_ingredients(query)
    measurement_request = asks_about_item_measurements(query)
    ingredient_amount_request = asks_about_ingredient_amount(query)
    if ingredient_request or measurement_request or ingredient_amount_request:
        matches = find_menu_matches(query, catalog) or list(context_items)
        if matches:
            rows: list[str] = []
            for item in matches:
                facts: list[str] = []
                if ingredient_request:
                    facts.append(
                        "ingredients: "
                        + (", ".join(item.ingredients) if item.ingredients else "not listed")
                    )
                if ingredient_amount_request:
                    facts.append(
                        "listed ingredients: "
                        + (", ".join(item.ingredients) if item.ingredients else "not listed")
                    )
                    facts.append("per-ingredient amounts are not listed in the catalog")
                if measurement_request:
                    measurements = listed_measurements(item.description)
                    facts.append(
                        "listed measurement: " + ", ".join(measurements)
                        if measurements
                        else "diameter in inches and ingredient or portion weights are not listed in the catalog"
                    )
                rows.append(f"- {item.name}: " + "; ".join(facts))
            return "\n".join(rows)
        return (
            "I could not match that item-detail question to one menu item. "
            "Ask with the pizza name; allergy or cross-contamination questions are passed to restaurant staff."
        )
    category = next((value for value in categories if normalize_text(value) in normalized.split()), None)
    if category:
        items = [item for item in catalog.active_items if item.category == category]
        return "\n".join(
            [f"**{category}**", *(f"- {item.name}: {format_money(item.price, catalog.currency)}" for item in items)]
        )
    if "menu" in normalized.split() or "show" in normalized.split():
        return "\n".join(
            f"**{category}:** " + ", ".join(item.name for item in catalog.active_items if item.category == category)
            for category in categories
        )
    matches = find_menu_matches(query, catalog)
    if matches:
        return "Here is what matched the catalog:\n" + "\n".join(
            f"- {item.name}: {format_money(item.price, catalog.currency)}" for item in matches
        )
    return "I can help with the menu, a draft order, changes, confirmation, cancellation, and billing."


def contains_any(text: str, words: Iterable[str]) -> bool:
    return bool(set(normalize_text(text).split()).intersection(words))
