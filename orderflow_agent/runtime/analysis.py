"""Operational performance summaries across agent-control modes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def assigned_condition(strictness: int) -> str:
    if strictness <= 33:
        return "flexible"
    if strictness <= 66:
        return "assisted"
    return "controlled"


def condition_summary(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("agent_mode") or row.get("prompt_mode") or assigned_condition(int(row["strictness"])))
        grouped[condition].append(row)
    summary = []
    for condition in ("controlled", "assisted", "flexible"):
        items = grouped.get(condition, [])
        if not items:
            continue
        repair_requests = sum(int(item.get("repair_requests", 0)) for item in items)
        successful_repairs = sum(int(item.get("successful_repairs", 0)) for item in items)
        successful = sum(int(item.get("confirmed_orders", 0)) > 0 for item in items)
        summary.append(
            {
                "agent_mode": condition,
                "sessions": len(items),
                "verified_task_success_rate": successful / len(items),
                "repair_requests": repair_requests,
                "repair_success_rate": successful_repairs / repair_requests if repair_requests else None,
                "compliance_failures": sum(int(item.get("compliance_failures", 0)) for item in items),
                "confirmed_orders": sum(int(item.get("confirmed_orders", 0)) for item in items),
            }
        )
    return summary
