# Safety and Limits

keept food safety and transaction authority outside model judgment.

Allergy, cross-contamination, food-poisoning, and other safety-sensitive requests are transferred when the catalog cannot verify the answer. The agent clearly says restaurant staff are needed. A conversation signal cannot override this rule, and a frustration signal alone cannot trigger a high-impact action.

Once a handover is pending, the same session cannot place, edit, confirm, or cancel an order through the automated agent. New customer messages are stored in the ticket for staff instead. A staff reply is shown in the customer chat, and completion clears the lock and records the final staff response; starting a separate conversation creates a separate session.

Catalog prices, quantities, totals, confirmation, and persistence are deterministic. The model cannot authorize refunds, discounts, compensation, delivery guarantees, or payments. A customer can retrieve locally persisted confirmation facts with the unique order reference, but the demo has no external fulfilment or payment-system status; complaints and payment disputes therefore go to the staff queue.

The message boundary rejects explicit and split-turn prompt replacement, fake system/developer/tool roles, secret-extraction requests, encoded-follow instructions, script payloads, and tested Unicode, multilingual, and low-effort obfuscation before any ordering tool runs. It limits customer messages to 1,200 characters, each item to 20 units, and a complete draft to 50 units. Quantity parsing rejects negative, fractional, arithmetic, non-finite, non-ASCII numeric, and ambiguous multi-quantity forms instead of silently converting them. These controls reduce tested abuse paths; they are not a claim that every possible adversarial phrase is detectable.

The response model receives no raw customer history or raw current wording. It receives a typed, fact-only operational brief and a narrow writing task. Complete replies are reviewed before display for transaction, cart, catalogue, measurement, confirmation, handover, and internal-label violations.

Delivery and pickup are deterministic draft fields. A delivery order cannot reach its confirmation gate without a supplied address, and a switch to pickup clears any draft address. The local demo stores that address in SQLite and order exports; a real deployment needs encryption, restricted staff access, retention limits, and deletion controls.

Address text is customer data, not a source of price or order state. The address validator canonicalises encoded, compatibility, confusable, and zero-width text before rejecting currency amounts, discounts, order references, transaction-status claims, and response-steering instructions. Response briefs read totals only from deterministic bill fields and remove transaction-like fragments from legacy address data, so an old poisoned address cannot replace the catalogue total in a confirmation or order lookup.

The final placement gate accepts an unambiguous yes or no. Repeating `confirm order`, saying `place order`, combining approval with a cart change, or supplying both delivery and pickup does not place or partially mutate an order. Mixed add/remove turns are split by asking the customer for one action at a time, and an attempted removal above the current cart quantity leaves the draft unchanged.

The conversation analyzer describes observable language and workflow events. Its labels are not psychological assessments. Source turns and evidence are stored so an operator can inspect the basis.

The menu-interest dataset and retrieval cases are synthetic. The generated menu photographs are illustrative.

The automated scenario runner verifies configured software outcomes only. It does not measure satisfaction, trust, intent, or causal effects.

keept the exploratory, regression, and optional live-model adversarial workflow in [ADVERSARIAL_TESTING.md](ADVERSARIAL_TESTING.md). The cases document tested boundaries and known corrections; they do not imply universal prompt-injection detection.

The in-application queue uses browser polling and local SQLite. It has no external employee notification, service-level timer, assignment policy, or concurrency controls for multiple staff members.

Before using customer data, I would add authentication, staff roles, encrypted storage, retention and deletion controls, a real staff-routing service, an audited payment boundary, store-system integration, monitoring, and a reviewed allergen-information process.
