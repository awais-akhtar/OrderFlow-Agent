# Data Dictionary

OrderFlow-Agent uses SQLite through `SQLiteStorageAdapter`. The default local path is `data/orderflow.db` and is ignored by Git.

## agent_sessions

- `id`: generated session identifier.
- `agent_mode`: controlled, assisted, or flexible.
- `strictness`: numeric mode representation.
- `turn_count`: customer turns processed.
- `repair_requests` and `successful_repairs`: observable correction counts.
- `compliance_failures`: blocked outward-response failures.
- `confirmed_orders`: persisted orders from the session.
- `failed_attempts`, `validation_failures`, `unsupported_attempts`, `confirmation_failures`: operational counters used by evaluation and escalation.
- `handover_active`: whether automated transaction actions are paused for this session.
- `handover_case_id`: the pending case responsible for that pause, or an empty string.
- `fulfilment`: undecided, delivery, or pickup for the active draft.
- `delivery_address`: customer-supplied address for an active delivery draft, otherwise empty.
- `started_at` and `updated_at`: UTC timestamps.

## conversation_turns

- `id` and `session_id`: turn and parent identifiers.
- `role`: customer, AI, or human.
- `content`: stored transcript text.
- `channel`: text or voice.
- `created_at`: UTC timestamp.

## tool_traces

- `session_id` and `user_turn_id`: lineage back to the triggering turn.
- `payload`: JSON list of tool names, status, detail, and timestamp.
- `created_at`: UTC timestamp.

## conversation_signals

- `label`: neutral, confused, frustrated, satisfied, or urgent.
- `confidence`: bounded support score, not a diagnosis.
- `evidence`: transparent derivation text.
- `source_turns_json`: source turn identifiers.
- `method`: deterministic or customer-provided derivation boundary.
- `created_at`: UTC timestamp.

## orders

- `id` and `session_id`: order and originating session.
- `status`: confirmed in the current workflow.
- `currency` and `total`: catalog-derived monetary values in integer units.
- `lines_json`: SKU, display name, quantity, unit price, and line total.
- `fulfilment`: delivery or pickup at confirmation time.
- `delivery_address`: stored only for a delivery order; pickup stores an empty string.
- `created_at`: UTC confirmation time.

## handovers

- `id` and `session_id`: case and conversation identifiers.
- `status`: pending or completed.
- `payload`: typed decision, signal, full conversation history, cart, tool history, actions, structured summary, live messages, human response, and explicitly carried cart facts.
- `live_messages`: ordered message objects containing an id, `customer` or `staff` role, content, and UTC creation time.
- `queue_position`: derived at read time from pending cases ordered by `created_at`; it is not stored as mutable ticket data.
- `human_response`: the final staff resolution message recorded when the case is completed.
- `context_carryover_score`: fraction of active cart facts explicitly selected by the operator; no active cart is displayed as not applicable in the UI.
- `created_at` and `updated_at`: UTC timestamps.

## knowledge_documents and knowledge_chunks

Documents retain title, safe source basename, MIME type, extracted text, checksum, and timestamp. Chunks retain parent id, title, source, exact chunk text, ordinal, and checksum. Checksums support duplicate detection and index generation.

## retrieval_traces

- `query`: the retrieval question.
- `provider_id`: generated-answer provider or local extractive path.
- `answer_generated`: whether generation followed retrieval.
- `trace_json`: lane names, candidate/selected counts, timings, fusion method, and non-secret cache key.
- `hit_ids_json`: ordered selected chunk identifiers.
- `created_at`: UTC timestamp.

The schema retains a `voice` channel for adapter compatibility, but the current Reflex customer page accepts text turns only. The provider boundary still supports completed-turn OpenAI transcription and does not persist raw audio.
