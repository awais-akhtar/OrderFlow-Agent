"""Operational records shared by the agent, tools, storage, and interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from .modes import AgentMode, mode_from_strictness


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


PendingAction = Literal[
    "none",
    "collect_fulfilment",
    "collect_delivery_address",
    "confirm_order",
    "confirm_cancel",
    "collect_order_reference",
    "handover",
]
FulfilmentType = Literal["undecided", "delivery", "pickup"]


@dataclass
class ConversationSession:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    order: dict[str, int] = field(default_factory=dict)
    pending_action: PendingAction = "none"
    strictness: int = 50
    turn_count: int = 0
    repair_requests: int = 0
    successful_repairs: int = 0
    compliance_failures: int = 0
    confirmed_orders: int = 0
    failed_attempts: int = 0
    validation_failures: int = 0
    unsupported_attempts: int = 0
    confirmation_failures: int = 0
    handover_active: bool = False
    handover_case_id: str = ""
    fulfilment: FulfilmentType = "undecided"
    delivery_address: str = ""
    last_user_message: str = ""
    menu_context: tuple[str, ...] = ()
    started_at: str = field(default_factory=utc_now)

    @property
    def agent_mode(self) -> AgentMode:
        return mode_from_strictness(self.strictness)

    @property
    def prompt_mode(self) -> AgentMode:
        """Backward-compatible alias for earlier strictness integrations."""
        return self.agent_mode


@dataclass(frozen=True)
class ToolStep:
    name: str
    status: Literal["passed", "blocked", "fallback", "info"] = "passed"
    detail: str = ""
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class BillLine:
    sku: str
    item: str
    quantity: int
    unit_price: int
    total: int


@dataclass(frozen=True)
class Bill:
    lines: tuple[BillLine, ...]
    grand_total: int
    currency: str


@dataclass(frozen=True)
class MenuAttachment:
    """Catalog-owned product details rendered alongside an agent reply."""

    sku: str
    title: str
    description: str
    ingredients: tuple[str, ...]
    image: str
    price: int
    currency: str


@dataclass(frozen=True)
class AgentResponse:
    content: str
    tool_trace: tuple[ToolStep, ...] = ()
    confirmed_order_id: str | None = None
    handover_requested: bool = False
    conversation_signal: Any | None = None
    handover_decision: Any | None = None
    handover_case_id: str | None = None
    menu_attachments: tuple[MenuAttachment, ...] = ()


@dataclass(frozen=True)
class TranscriptTurn:
    session_id: str
    role: Literal["customer", "ai", "human"]
    content: str
    channel: Literal["text", "voice"] = "text"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class OrderRecord:
    session_id: str
    lines: tuple[BillLine, ...]
    total: int
    currency: str
    fulfilment: Literal["delivery", "pickup"] = "pickup"
    delivery_address: str = ""
    status: Literal["confirmed", "cancelled"] = "confirmed"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
