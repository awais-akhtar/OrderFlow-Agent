"""Handover orchestration over policy, context, and persisted history."""

from __future__ import annotations

from dataclasses import asdict

from orderflow_agent.context.models import ConversationSignal
from orderflow_agent.models import ConversationSession

from .models import HandoverCase, HandoverDecision
from .policy import HandoverPolicy
from .summary import build_handover_summary


class HandoverService:
    def __init__(self, storage, policy: HandoverPolicy | None = None) -> None:
        self.storage = storage
        self.policy = policy or HandoverPolicy()

    def decide(
        self,
        message: str,
        session: ConversationSession,
        signal: ConversationSignal,
    ) -> HandoverDecision:
        return self.policy.decide(message, session, signal)

    def create_case(
        self,
        *,
        session: ConversationSession,
        decision: HandoverDecision,
        signal: ConversationSignal,
        customer_request: str,
        issue: str | None = None,
    ) -> HandoverCase:
        turns = self.storage.list_turns(session.session_id)
        traces = self.storage.list_tool_traces(session.session_id)
        actions = tuple(
            step.get("name", "")
            for trace in reversed(traces)
            for step in trace.get("steps", [])
            if step.get("status") == "passed" and step.get("name")
        )[-10:]
        context = (
            f"Customer appears {signal.label} based on observable conversation events: {signal.evidence}"
            if signal.label != "neutral"
            else signal.evidence
        )
        outstanding = issue or decision.reason
        next_action = self._suggested_action(decision.trigger)
        summary = build_handover_summary(
            customer_request=customer_request,
            issue=decision.reason,
            cart=dict(session.order),
            actions_attempted=actions,
            outstanding_problem=outstanding,
            relevant_customer_context=context,
            suggested_next_action=next_action,
            fulfilment=session.fulfilment,
            delivery_address=session.delivery_address,
        )
        case = HandoverCase(
            session_id=session.session_id,
            decision=decision,
            signal=signal,
            customer_request=customer_request,
            issue=decision.reason,
            cart=dict(session.order),
            conversation_history=tuple(asdict(turn) for turn in turns),
            tool_history=tuple(traces),
            actions_attempted=actions,
            outstanding_problem=outstanding,
            relevant_customer_context=context,
            suggested_next_action=next_action,
            summary=summary,
            fulfilment=session.fulfilment,
            delivery_address=session.delivery_address,
        )
        self.storage.save_handover(case)
        return case

    @staticmethod
    def _suggested_action(trigger: str) -> str:
        return {
            "refund_or_payment_dispute": "Verify the payment record and apply the approved refund or dispute process.",
            "allergy_or_food_safety": "Confirm ingredients and cross-contamination controls with an authorised employee.",
            "existing_order_complaint": "Locate the existing order and review the complaint against store policy.",
            "repeated_misunderstanding": "Restate the requested change and update the cart with the customer present.",
            "repeated_validation_failures": "Review the cart and validation evidence before attempting another change.",
            "unsupported_request": "Explain the supported channel or route the request to the responsible team.",
        }.get(trigger, "Review the preserved conversation and resolve the outstanding request with the customer.")
