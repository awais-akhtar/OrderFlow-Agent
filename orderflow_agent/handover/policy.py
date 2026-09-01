"""Auditable pizza-order escalation rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from orderflow_agent.context.models import ConversationSignal
from orderflow_agent.models import ConversationSession

from .models import HandoverDecision


EXPLICIT_HUMAN = re.compile(
    r"\b(?:speak|talk|chat)\b.{0,18}\b(?:to|with)\b.{0,18}"
    r"\b(?:human|person|(?:real|actual)\s+person|manager|supervisor|operator|staff|representative|employe{0,2}s?|team\s+member)\b"
    r"|\b(?:transfer|connect|escalate|bring|call|get|put|pass)\b.{0,45}"
    r"\b(?:human|person|(?:real|actual)\s+person|manager|supervisor|operator|staff|representative|employe{0,2}s?|team\s+member)\b"
    r"|\b(?:want|need|request|prefer|require)\b.{0,32}"
    r"\b(?:human|person|(?:real|actual)\s+person|manager|supervisor|operator|staff|representative|employe{0,2}s?|team\s+member|live\s+agent)\b"
    r"|\b(?:member\s+of\s+staff|staff\s+member)\b"
    r"|\b(?:human|(?:real|actual)\s+person|manager|supervisor|operator|staff|representative|employe{0,2}s?|live\s+agent)\b"
    r"\s+(?:available|please|now|support|help|assistance|here)\b",
    re.IGNORECASE,
)
INDIRECT_HUMAN = re.compile(
    r"\b(?:put|pass|connect|transfer|get)\s+me\s+(?:through|over|connected)?\b.{0,35}"
    r"\b(?:someone|somebody)\b.{0,35}\b(?:restaurant|team|customer\s+(?:service|support))\b"
    r"|\b(?:someone|somebody)\b.{0,30}\b(?:at|from|on)\b.{0,18}\b(?:restaurant|team|customer\s+(?:service|support))\b"
    r"|\b(?:customer\s+(?:service|support)|live\s+agent|(?:real|actual)\s+person)\b",
    re.IGNORECASE,
)
NEGATED_HUMAN = re.compile(
    r"\b(?:do\s+not|don't|dont|not|never)\s+"
    r"(?:(?:want|need|request|require|prefer|have)\s+)?(?:a|an|any)?\s*"
    r"(?:human|person|manager|supervisor|operator|staff|representative|employe{0,2}s?|live\s+agent|team\s+member)\b"
    r"|\b(?:do\s+not|don't|dont|never)\s+(?:want|need|prefer)\s+to\s+"
    r"(?:speak|talk|chat)\s+(?:to|with)\s+"
    r"(?:a|an|any)?\s*(?:human|person|manager|supervisor|operator|staff|representative|employe{0,2}s?|team\s+member)\b"
    r"|\b(?:do\s+not|don't|dont|never)\s+"
    r"(?:transfer|connect|escalate|bring|call|get|put|pass)\b.{0,28}"
    r"\b(?:human|person|manager|supervisor|operator|staff|representative|employe{0,2}s?|team\s+member)\b",
    re.IGNORECASE,
)
STAFF_AUDIENCE_CONTEXT = re.compile(
    r"\bstaff\b.{0,16}\b(?:discount|meal|menu|order|party|pizza|special)\b",
    re.IGNORECASE,
)
REFUND_PAYMENT = (
    "refund", "money back", "charged twice", "double charged", "payment dispute", "payment failed",
    "payment declined", "wrong charge", "card charged", "unrecognised charge", "unrecognized charge",
)
ALLERGY_SAFETY = (
    "allergy", "allergic", "anaphyl", "cross contamination", "cross-contamination", "food poisoning", "unsafe to eat",
)
EXISTING_ORDER_COMPLAINT = (
    "my order is late", "order never arrived", "wrong order", "missing item", "food is cold", "complaint",
    "driver never", "existing order", "arrived burnt", "food was burnt", "undercooked", "soggy pizza",
    "didn't get it", "did not get it", "didn't receive", "did not receive", "not received", "never got it",
)
UNSUPPORTED_REQUESTS = (
    "book a table", "reserve a table", "catering contract", "job application", "franchise application",
)


@dataclass(frozen=True)
class HandoverPolicy:
    failed_attempt_threshold: int = 2
    validation_failure_threshold: int = 2

    def decide(
        self,
        message: str,
        session: ConversationSession,
        signal: ConversationSignal,
    ) -> HandoverDecision:
        text = message.casefold()
        if self._requests_human(message):
            return self._yes("explicit_human_request", "The customer explicitly requested restaurant staff.", 1.0)
        if self._contains(text, REFUND_PAYMENT):
            return self._yes("refund_or_payment_dispute", "Refund and payment disputes require restaurant staff.", 0.98)
        if self._contains(text, ALLERGY_SAFETY):
            return self._yes(
                "allergy_or_food_safety",
                "The request involves allergy or food-safety information that this catalog cannot verify confidently.",
                0.99,
            )
        if self._contains(text, EXISTING_ORDER_COMPLAINT):
            return self._yes("existing_order_complaint", "The customer reported a problem with an existing order.", 0.96)
        if self._contains(text, UNSUPPORTED_REQUESTS):
            return self._yes("unsupported_request", "The request is outside this ordering agent's supported workflow.", 0.92)
        if session.validation_failures >= self.validation_failure_threshold:
            return self._yes(
                "repeated_validation_failures",
                f"The ordering workflow reached {session.validation_failures} validation failures.",
                0.9,
            )
        if session.failed_attempts >= self.failed_attempt_threshold:
            return self._yes(
                "repeated_misunderstanding",
                f"The conversation reached {session.failed_attempts} failed resolution attempts.",
                0.88,
            )
        if signal.label == "frustrated":
            return HandoverDecision(
                False,
                "A frustration signal was recorded as supporting context, but it is not used alone to force a handover.",
                signal.confidence,
                "supporting_conversation_signal",
            )
        return HandoverDecision(False, "No escalation rule matched.", 0.95, "none")

    @staticmethod
    def _contains(text: str, phrases: tuple[str, ...]) -> bool:
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _requests_human(message: str) -> bool:
        negated_spans = [match.span() for match in NEGATED_HUMAN.finditer(message)]
        audience_spans = [match.span() for match in STAFF_AUDIENCE_CONTEXT.finditer(message)]
        for pattern in (EXPLICIT_HUMAN, INDIRECT_HUMAN):
            for match in pattern.finditer(message):
                start, end = match.span()
                overlaps_negation = any(start < other_end and end > other_start for other_start, other_end in negated_spans)
                overlaps_audience = any(start < other_end and end > other_start for other_start, other_end in audience_spans)
                if not overlaps_negation and not overlaps_audience:
                    return True
        return False

    @staticmethod
    def _yes(trigger: str, reason: str, confidence: float) -> HandoverDecision:
        return HandoverDecision(True, reason, confidence, trigger)
