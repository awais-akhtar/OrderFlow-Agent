"""Conservative supporting context derived from observable conversation events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


SignalLabel = Literal["neutral", "confused", "frustrated", "satisfied", "urgent"]


@dataclass(frozen=True)
class ConversationSignal:
    label: SignalLabel
    confidence: float
    evidence: str
    source_turns: tuple[str, ...]
    method: str = "deterministic-observable-events"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Conversation signal confidence must be between 0 and 1.")
