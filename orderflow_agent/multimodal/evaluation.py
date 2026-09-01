"""Reproducible retrieval metrics for the synthetic pizza-menu cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from orderflow_agent.catalog import JsonCatalogStore

from .intelligence import MenuIntelligence


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MenuRetrievalReport:
    case_count: int
    top_k: int
    recall_at_k: float
    mean_reciprocal_rank: float
    cases: tuple[dict[str, Any], ...]
    dataset_label: str


def evaluate_menu_retrieval(
    case_path: str | Path = ROOT / "evaluation" / "menu_retrieval_cases.json",
    *,
    top_k: int = 3,
) -> MenuRetrievalReport:
    payload = json.loads(Path(case_path).read_text(encoding="utf-8"))
    if payload.get("dataset_label") not in {"synthetic", "synthetic-demo", "demo"}:
        raise ValueError("Menu retrieval evaluation requires a clearly labelled synthetic/demo case set.")
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("Menu retrieval evaluation needs at least one case.")
    top_k = max(1, min(int(top_k), 20))
    intelligence = MenuIntelligence(JsonCatalogStore().load(), asset_root=ROOT)
    rows: list[dict[str, Any]] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        expected = set(case["relevant_skus"])
        recommendations = intelligence.recommend(case["query"], limit=top_k)
        ranked = [row.item.sku for row in recommendations]
        retrieved = expected.intersection(ranked)
        recall = len(retrieved) / len(expected)
        first_rank = next((index for index, sku in enumerate(ranked, start=1) if sku in expected), None)
        reciprocal_rank = 1 / first_rank if first_rank else 0.0
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected_skus": sorted(expected),
                "ranked_skus": ranked,
                "recall_at_k": round(recall, 6),
                "reciprocal_rank": round(reciprocal_rank, 6),
            }
        )
    return MenuRetrievalReport(
        case_count=len(rows),
        top_k=top_k,
        recall_at_k=round(sum(recalls) / len(recalls), 6),
        mean_reciprocal_rank=round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6),
        cases=tuple(rows),
        dataset_label=str(payload["dataset_label"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OrderFlow pizza-menu retrieval.")
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation" / "menu_retrieval_cases.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = evaluate_menu_retrieval(arguments.cases, top_k=arguments.top_k)
    rendered = json.dumps(asdict(report), indent=2, ensure_ascii=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
