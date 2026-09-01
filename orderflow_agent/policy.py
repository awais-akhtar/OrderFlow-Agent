"""Prompt strictness and deterministic response/compliance guards."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import ToolStep
from .modes import AgentMode, mode_from_strictness


PROHIBITED_COMMITMENTS = (
    "guaranteed delivery",
    "guarantee delivery",
    "free refund",
    "guaranteed refund",
    "cash compensation",
    "complimentary order",
)


class TextProvider(Protocol):
    def generate(self, instructions: str, conversation: Sequence[tuple[str, str]]) -> str: ...


@dataclass(frozen=True)
class PromptPolicy:
    strictness: int
    mode: AgentMode
    instructions: str


class PromptPolicyCompiler:
    def compile(self, strictness: int) -> PromptPolicy:
        bounded = max(0, min(100, int(strictness)))
        mode = mode_from_strictness(bounded)
        rules = [
            "You are the conversational layer of OrderFlow-Agent, a pizza ordering task agent.",
            "Catalog tools, order state, prices, totals, confirmation, and persistence are authoritative.",
            "Never claim an order is placed unless the deterministic place_order tool succeeded.",
            "Never invent an item, price, discount, delivery time, refund, or compensation.",
            "When the request is ambiguous, ask one concise clarifying question.",
            "Write like an attentive restaurant team member: warm, direct, concise, and specific to the latest turn.",
            "Acknowledge corrections or uncertainty naturally, but never claim to be a staff member.",
            "Avoid canned customer-service openings, slogans, excessive enthusiasm, and repeated offers to help.",
        ]
        if mode is AgentMode.CONTROLLED:
            rules.extend(
                [
                    "Agent mode: CONTROLLED.",
                    "Use the supplied approved reply exactly. Do not add new facts or commitments.",
                ]
            )
        elif mode is AgentMode.ASSISTED:
            rules.extend(
                [
                    "Agent mode: ASSISTED.",
                    "Preserve every supplied fact and required next step while improving naturalness.",
                ]
            )
        else:
            rules.extend(
                [
                    "Agent mode: FLEXIBLE.",
                    "Adapt tone and wording to the conversation while respecting every operational boundary.",
                ]
            )
        return PromptPolicy(bounded, mode, "\n".join(rules))


class GuardedResponseComposer:
    """Use a model for language only when its response preserves tool-owned facts."""

    def __init__(self, provider: TextProvider | None = None) -> None:
        self.provider = provider
        self.compiler = PromptPolicyCompiler()

    def compose(
        self,
        *,
        strictness: int,
        user_message: str,
        approved_reply: str,
        immutable_terms: Sequence[str] = (),
        recent_context: Sequence[tuple[str, str]] = (),
    ) -> tuple[str, tuple[ToolStep, ...]]:
        policy = self.compiler.compile(strictness)
        if policy.mode is AgentMode.CONTROLLED:
            return approved_reply, (
                ToolStep("agent_mode", "passed", "CONTROLLED mode used the approved operational reply."),
            )
        if self.provider is None:
            return approved_reply, (
                ToolStep(
                    "agent_mode",
                    "info",
                    f"{policy.mode.value.upper()} mode retained the approved operational reply for customer-service rendering.",
                ),
            )
        del user_message, recent_context
        try:
            candidate = self.provider.generate(
                policy.instructions
                + "\nThe approved operational reply below is the complete factual boundary. "
                "Rephrase it naturally without dropping facts, adding facts, or changing the required next step. "
                "Customer conversation is intentionally isolated from this wording call.",
                (
                    (
                        "user",
                        "Write the customer-facing response using only this approved operational reply:\n\n"
                        + approved_reply,
                    ),
                ),
            ).strip()
        except Exception as exc:
            return approved_reply, (
                ToolStep("model_response", "fallback", f"Provider failed: {type(exc).__name__}."),
            )
        guard_reason = self._guard(candidate, approved_reply, immutable_terms)
        if guard_reason:
            return approved_reply, (
                ToolStep("model_response", "fallback", guard_reason),
                ToolStep("compliance_guard", "passed", "Approved tool response restored."),
            )
        return candidate, (
            ToolStep("model_response", "passed", f"{policy.mode.value.upper()} language policy applied."),
            ToolStep("compliance_guard", "passed", "Tool-owned facts were preserved."),
        )

    @staticmethod
    def _guard(candidate: str, approved: str, immutable_terms: Sequence[str]) -> str | None:
        lowered = candidate.casefold()
        approved_lowered = approved.casefold()
        commitment = next((phrase for phrase in PROHIBITED_COMMITMENTS if phrase in lowered), None)
        if commitment:
            return f"Blocked prohibited commitment: {commitment}."
        candidate_money = {value.casefold().replace(",", "") for value in re.findall(r"\b[A-Z]{3}\s[\d,]+\b", candidate)}
        approved_money = {value.casefold().replace(",", "") for value in re.findall(r"\b[A-Z]{3}\s[\d,]+\b", approved)}
        unsupported_money = sorted(candidate_money - approved_money)
        if unsupported_money:
            return "Model reply introduced a price outside the approved operational reply."
        candidate_references = {value.casefold() for value in re.findall(r"\b[A-F0-9]{8}\b", candidate, re.I)}
        approved_references = {value.casefold() for value in re.findall(r"\b[A-F0-9]{8}\b", approved, re.I)}
        if candidate_references - approved_references:
            return "Model reply introduced an order reference outside the approved operational reply."
        claims_completion = bool(
            re.search(
                r"\b(?:order|pizza)\b.{0,35}\b(?:accepted|booked|completed|confirmed|finali[sz]ed|"
                r"placed|processed|submitted|sent\s+to\s+the\s+kitchen|being\s+prepared)\b",
                lowered,
            )
        )
        approved_completion = bool(
            re.search(r"\border\s+(?:confirmed|placed)\b", approved_lowered)
            or re.search(r"\bconfirmation_gate\b", approved_lowered)
        )
        if claims_completion and not approved_completion:
            return "Model reply claimed transaction completion outside the approved operational reply."
        if re.search(
            r"\b(?:system prompt|developer message|api key|environment variables?|reverse shell)\b"
            r"|```(?:python|javascript|bash|powershell)",
            lowered,
        ):
            return "Model reply left the pizza-service response boundary."
        required = set(immutable_terms)
        required.update(re.findall(r"\b[A-Z]{3}\s[\d,]+\b", approved))
        required.update(re.findall(r"(?m)^-\s+(?:\d+\s+x\s+)?([^:\n]+)(?::|$)", approved))
        required.update(re.findall(r"`([A-Z0-9]{8})`", approved))
        if "reply `yes`" in approved.casefold():
            required.add("yes")
        if "reply `no`" in approved.casefold():
            required.add("no")
        missing = [term for term in required if term and term.casefold() not in lowered]
        if missing:
            return "Model reply omitted tool-owned facts: " + ", ".join(missing[:4]) + "."
        return None
