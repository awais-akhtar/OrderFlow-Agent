# Agent Behaviour Evaluation

Used deterministic scenario expectations to check the ordering software, not to infer customer experience. The default configuration in `evaluation/configs/default.json` replays 18 scenarios across CONTROLLED, ASSISTED, and FLEXIBLE modes with seed 42, producing 54 mode/scenario runs.

## Run

```powershell
.\.venv\Scripts\python -m orderflow_agent.evaluation.runner
```

Use a separate output directory when needed:

```powershell
.\.venv\Scripts\python -m orderflow_agent.evaluation.runner --output .\evaluation\results
```

The runner exits with code 0 only when every configured expectation check passes.

## Reproducible Artifacts

Each timestamped run stores the exact configuration and scenario set beside JSON, CSV, and Markdown results. Metadata records the provider, model, agent modes, random seed, timestamp, package/catalog and dependency versions, Python/platform version, and Git commit when available.

The current adapter exposes no common token-usage contract, so `token_usage` is `null` unless that boundary is added. The model name remains in run metadata. Latency is measured around each complete agent turn and therefore includes configured provider time when a provider is active.

## Interpretation

`successful_task_completion` means the scenario's machine-checkable expectations passed. It does not mean a human liked the conversation. Unsupported item attempts count blocked extraction calls, handovers count matched escalation decisions, and confirmation failures count unclear responses while a confirmation gate is open.

The hard cases include conflicting confirmation and mutation text, conflicting cancellation text, duplicate confirmation, repeated validation failures, unsupported service requests, and several operational handover triggers. They verify software invariants; they are not field observations.

## Menu Retrieval

The separate synthetic menu case set checks Recall@K and mean reciprocal rank:

```powershell
.\.venv\Scripts\python -m orderflow_agent.multimodal.evaluation --top-k 3 --output .\evaluation\results\menu-retrieval.json
```

`evaluation/menu_retrieval_cases.json` is explicitly labelled `synthetic-demo`.
