# OrderFlow operating knowledge

OrderFlow-Agent is a conversational ordering application. Natural language stays flexible, while catalog lookup, draft mutation, confirmation, billing, and order persistence run through deterministic tools.

## Ordering rules

- Only active catalog items can be added to a draft.
- Catalog prices are the source of truth.
- A customer can change the draft before confirmation.
- Placing an order requires a separate explicit yes response after the bill is shown.
- Cancelling a non-empty draft also requires confirmation.
- The demo does not promise delivery times, refunds, discounts, or compensation.

## handover

The handover brief carries the observable conversation signal, current order facts, completed tool actions, and unresolved issue. The operator can record which cart facts were carried into the conversation. Conversation signals are supporting context, not a diagnosis or a reason for high-impact action on their own.

## Model boundary

An optional model provider may interpret or polish language. It cannot alter catalog items, quantities, prices, totals, or the confirmation state. When model output fails a deterministic guard, the application uses its approved response instead and records that fallback in the tool trace.
