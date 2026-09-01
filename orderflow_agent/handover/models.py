"""Typed handover decisions and preserved operator cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from orderflow_agent.context.models import ConversationSignal


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class HandoverDecision:
    should_handover: bool
    reason: str
    confidence: float
    trigger: str
    timestamp: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Handover confidence must be between 0 and 1.")


@dataclass(frozen=True)
class HandoverCase:
    session_id: str
    decision: HandoverDecision
    signal: ConversationSignal
    customer_request: str
    issue: str
    cart: dict[str, int]
    conversation_history: tuple[dict[str, Any], ...]
    tool_history: tuple[dict[str, Any], ...]
    actions_attempted: tuple[str, ...]
    outstanding_problem: str
    relevant_customer_context: str
    suggested_next_action: str
    summary: str
    fulfilment: str = "undecided"
    delivery_address: str = ""
    status: Literal["pending", "completed"] = "pending"
    human_response: str = ""
    facts_carried_forward: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=_now)

    @property
    def context_carryover_score(self) -> float:
        """Report how much of the current cart the operator explicitly retained."""
        cart_facts = {f"{quantity} x {name}" for name, quantity in self.cart.items()}
        if not cart_facts:
            return 1.0
        return round(len(cart_facts.intersection(self.facts_carried_forward)) / len(cart_facts), 3)
