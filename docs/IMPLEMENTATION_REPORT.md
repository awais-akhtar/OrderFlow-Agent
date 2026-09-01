# Implementation Report

I kept OrderFlow-Agent focused on pizza and food ordering. This report records what is implemented in the repository, what I verified on the development host, and what still depends on an external service or deployment decision.

## Fully Implemented

- `app.py` launches a Reflex application with a focused customer route at `/` and a separate restaurant workspace at `/staff`.
- The customer route streams assistant wording from the configured provider. When no streaming model is available, the UI shows a service notice instead of presenting a prewritten assistant reply as model output.
- Deterministic code owns catalogue matching, quantities, cart changes, prices, totals, fulfilment, delivery addresses, confirmation gates, order persistence, and handover locks.
- Delivery orders collect and validate an address before the confirmation gate. Pickup orders never retain a delivery address.
- Stream guards reject invented catalogue items, unsupported ingredients, altered prices or currencies, premature confirmation claims, mismatched order references, prohibited commitments, and off-topic replies.
- Customer chat is limited to two concise sentences per assistant turn. Relevant catalogue cards provide images, descriptions, prices, small ingredient lists, and deterministic add controls outside the generated prose.
- Handover decisions preserve the transcript, current cart, fulfilment state, tool history, actions already attempted, unresolved issue, supporting conversation signal, and a structured employee summary. A pending handover locks automated order changes, exposes a customer queue number, and routes new customer messages to the staff ticket until an employee closes it.
- Conversation signals are conservative operational observations with confidence, evidence, source turns, and derivation method. A frustration signal by itself cannot trigger handover.
- The JSON catalogue supports optional images, descriptions, ingredients, categories, and labelled interaction statistics. Menu intelligence combines text with optional image features, falls back to text-only retrieval, and supports pluggable embeddings.
- CONTROLLED, ASSISTED, and FLEXIBLE modes use different prompting freedom while sharing the same transaction, confirmation, pricing, and safety constraints.
- The evaluation runner writes configuration, scenario set, JSON, CSV, Markdown, seed, provider/model, timestamp, software version, dependency, platform, catalogue, and Git metadata.
- SQLite storage persists sessions, turns, tool traces, signals, orders, handovers, documents, chunks, and retrieval traces. Staff can export order history as CSV or JSON.
- The section-based staff route contains order history, a ticket inbox with customer/staff messages and resolution controls, catalogue editing, menu ranking, scenario evaluation, dual-lane RAG, tool traces, and runtime settings.
- The Reflex theme now gives the customer chat, menu cards, cart, queue state, staff navigation, metrics, tables, and ticket workspace responsive interaction states and reduced-motion support without changing their deterministic event handlers.
- The adversarial developer tooling includes a state-inspection probe, a permanent three-mode regression matrix, and an optional configured-model pass. The documented corpus covers input, state, transaction, handover, domain, catalogue identity, and model-output boundaries.
- KernelLoom Python/HTTP, OpenAI, hosted Hugging Face, and OpenAgent adapters remain explicit behind the provider registry. The customer runtime currently supports streamed KernelLoom, OpenAI, and hosted Hugging Face replies when configured.

## Verified

- Default test discovery ran 189 tests successfully; the opt-in browser test was skipped in that command as designed.
- The real Chromium test passed separately with `ORDERFLOW_RUN_BROWSER_E2E=1`. It exercised deterministic menu-card addition, a chat-triggered support ticket, customer and staff ticket messages, queue display, staff resolution, the disabled-provider service boundary, every staff navigation target, the side-by-side ticket workspace, mobile overflow, and a zero-error browser console.
- Reflex production export compiled 23 application units and completed all four production build stages.
- I exercised KernelLoom `0.4.1` from PyPI with local Qwen2.5 1.5B Instruct INT4 OpenVINO weights. A four-turn live stream listed the real catalogue families, added a Large Garden Special for delivery, requested the address, presented the PKR 2,199 confirmation gate, and confirmed exactly one persisted order with the supplied address.
- I also exercised the running Reflex customer page. Clicking the Small Cheese Pizza menu action updated the deterministic cart and streamed: "Would you like your Small Cheese Pizza delivered or would you prefer pickup?" The provider remained available, the browser console was clear, and the 390-pixel view had no horizontal overflow.
- The automated behavior evaluation passed 54/54 expectation checks across 18 scenarios and three modes.
- The adversarial matrix passed 297/297 checks: 99 isolated conversations in each of the three agent modes. The configured local model also passed the selected live-model adversarial set after deterministic handling.
- I reproduced the ordinary `aniversity special pizza` failure against the configured local model. Catalogue matching now requires distinctive item evidence, exact family phrases outrank shared words, unmatched items cannot populate menu context, and ingredient follow-ups without a valid item ask for the listed pizza name instead of guessing.
- The five-case dataset labelled `synthetic-demo` measured Recall@3 `1.0` and mean reciprocal rank `1.0`.

The scenario and retrieval results describe checked software expectations and a small synthetic retrieval set. They do not measure satisfaction, trust, psychological state, or causal effects.

## Partial Or External

- Local model weights are not shipped or downloaded. KernelLoom requires compatible model files and a supported local backend on the operator's machine.
- The bundled image representation is a transparent lightweight feature extractor, not a vision-language model.
- Handover is an in-application SQLite queue with browser polling. It does not contact a call centre, external ticketing system, or employee notification service.
- The customer route is text chat. Provider transcription and speech methods remain adapter capabilities but are not exposed as browser voice controls.
- The repository has no real payment processing, delivery dispatch, store inventory, staff authentication, role authorization, or allergen guarantee.

