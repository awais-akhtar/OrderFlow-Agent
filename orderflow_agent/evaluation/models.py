"""Configuration and result records for scenario replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    turns: tuple[str, ...]
    expected: dict[str, Any]


@dataclass(frozen=True)
class ScenarioMetrics:
    successful_task_completion: bool
    turns: int
    tool_calls: int
    validation_failures: int
    corrections_retries: int
    unsupported_item_attempts: int
    handovers: int
    confirmation_failures: int
    latency_ms: float
    mean_turn_latency_ms: float
    token_usage: int | None = None


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    description: str
    agent_mode: str
    provider: str
    passed: bool
    expectation_checks: dict[str, bool]
    metrics: ScenarioMetrics
    final_cart: dict[str, int]
    confirmed_order_ids: tuple[str, ...]
    handover_triggers: tuple[str, ...]
    assistant_responses: tuple[str, ...] = field(repr=False)
