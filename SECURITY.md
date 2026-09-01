# Security

OrderFlow-Agent loads credentials from environment variables or accepts them in the Settings screen for the current server-side workspace. `RuntimeSettings.safe_summary()` omits secret values, and SQLite records, exports, RAG traces, and evaluation.

The application rejects credentials embedded in provider URLs. OpenAI and Hugging Face are external processing boundaries. KernelLoom and OpenAgent default to loopback URLs, but loopback alone is not authentication.

Document ingestion accepts only text, Markdown, CSV, JSON, and PDF, limits each upload to 12 MB, strips path components from source names, and never executes uploaded content. Extracted documents, transcripts, order history, and handover records can still contain personal data.

The Catalog and Handovers screens have no login or role checks in this local demo. A shared deployment needs authentication, role-based authorization, audit logs, CSRF-aware deployment settings, encrypted storage and transport, retention/deletion rules, backups, and rate limits. Real payment credentials must remain outside this application unless an audited payment-provider integration is added.
