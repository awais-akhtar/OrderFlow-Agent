"""Deterministic conversation-signal fallback for operator context."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import ConversationSignal


FRUSTRATION_PHRASES = (
    "this is frustrating",
    "i am frustrated",
    "you keep getting",
    "not listening",
    "terrible service",
    "ridiculous",
    "fed up",
)
CONFUSION_PHRASES = (
    "i do not understand",
    "i don't understand",
    "what do you mean",
    "that is not what i meant",
    "you misunderstood",
    "i meant",
    "not that",
)
SATISFIED_PHRASES = ("thank you", "thanks", "perfect", "great", "that works", "looks good")
URGENT_PHRASES = ("urgent", "as soon as possible", "right now", "immediately", "emergency")


class ConversationContextAnalyzer:
    """Produces one supporting signal without treating it as a diagnosis."""

    def analyze(
        self,
        turns: Sequence[Any],
        *,
        failed_attempts: int = 0,
        validation_failures: int = 0,
        repair_requests: int = 0,
    ) -> ConversationSignal:
        customer_turns = [turn for turn in turns if self._role(turn) in {"customer", "user"}]
        recent = customer_turns[-4:]
        text = "\n".join(self._content(turn).casefold() for turn in recent)
        source_ids = tuple(self._id(turn, index) for index, turn in enumerate(recent, start=1))

        if self._contains(text, URGENT_PHRASES):
            return ConversationSignal(
                "urgent",
                0.8,
                "The customer used an explicit urgency phrase.",
                source_ids,
            )
        if self._contains(text, FRUSTRATION_PHRASES):
            confidence = min(0.9, 0.68 + 0.06 * min(failed_attempts + repair_requests, 3))
            return ConversationSignal(
                "frustrated",
                confidence,
                "The customer used an explicit frustration phrase"
                + (f" after {failed_attempts} failed attempt(s)." if failed_attempts else "."),
                source_ids,
            )
        if repair_requests or self._contains(text, CONFUSION_PHRASES):
            confidence = min(0.88, 0.62 + 0.07 * min(repair_requests + validation_failures, 3))
            return ConversationSignal(
                "confused",
                confidence,
                f"Observed {repair_requests} correction request(s) and {validation_failures} validation failure(s).",
                source_ids,
            )
        if self._contains(text, SATISFIED_PHRASES):
            return ConversationSignal(
                "satisfied",
                0.72,
                "The customer used an explicit positive completion phrase.",
                source_ids,
            )
        return ConversationSignal(
            "neutral",
            0.45,
            "No explicit confusion, frustration, satisfaction, or urgency cue was observed.",
            source_ids,
        )

    @staticmethod
    def _contains(text: str, phrases: Sequence[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _role(turn: Any) -> str:
        return str(turn.get("role", "") if isinstance(turn, dict) else getattr(turn, "role", ""))

    @staticmethod
    def _content(turn: Any) -> str:
        return str(turn.get("content", "") if isinstance(turn, dict) else getattr(turn, "content", ""))

    @staticmethod
    def _id(turn: Any, fallback: int) -> str:
        if isinstance(turn, dict):
            return str(turn.get("id") or f"turn-{fallback}")
        return str(getattr(turn, "id", f"turn-{fallback}"))
