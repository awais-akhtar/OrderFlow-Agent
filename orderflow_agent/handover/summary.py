"""Structured, concise summaries for restaurant staff."""

from __future__ import annotations

from collections.abc import Sequence


def build_handover_summary(
    *,
    customer_request: str,
    issue: str,
    cart: dict[str, int],
    actions_attempted: Sequence[str],
    outstanding_problem: str,
    relevant_customer_context: str,
    suggested_next_action: str,
    fulfilment: str = "undecided",
    delivery_address: str = "",
) -> str:
    cart_text = ", ".join(f"{quantity} x {name}" for name, quantity in cart.items()) or "No items in cart"
    actions = ", ".join(actions_attempted) or "No tool actions completed"
    return (
        f"Customer request: {customer_request}\n"
        f"Issue: {issue}\n"
        f"Current order/cart: {cart_text}\n"
        f"Fulfilment: {fulfilment}"
        + (f"; delivery address: {delivery_address}" if delivery_address else "")
        + "\n"
        f"Actions already attempted: {actions}\n"
        f"Outstanding problem: {outstanding_problem}\n"
        f"Relevant customer context: {relevant_customer_context}\n"
        f"Suggested next action: {suggested_next_action}"
    )
