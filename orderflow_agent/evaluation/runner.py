"""CLI and library runner for repeatable pizza-order scenarios."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import random
import subprocess
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orderflow_agent import __version__
from orderflow_agent.agent import ConversationalTaskAgent
from orderflow_agent.catalog import JsonCatalogStore
from orderflow_agent.modes import AgentMode, coerce_mode
from orderflow_agent.runtime.providers import ProviderRegistry, RuntimeSettings
from orderflow_agent.storage import SQLiteStorageAdapter

from .models import Scenario, ScenarioMetrics, ScenarioResult


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "evaluation" / "configs" / "default.json"


def load_scenarios(path: str | Path) -> list[Scenario]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Scenario(
            id=row["id"],
            description=row["description"],
            turns=tuple(row["turns"]),
            expected=dict(row.get("expected", {})),
        )
        for row in payload["scenarios"]
    ]


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scenario_path = (config_path.parent / config["scenario_set"]).resolve()
    scenarios = load_scenarios(scenario_path)
    seed = int(config.get("random_seed", 42))
    random.seed(seed)
    modes = [coerce_mode(value) for value in config.get("agent_modes", [mode.value for mode in AgentMode])]
    provider = str(config.get("provider", "disabled"))
    model = str(config.get("model", ""))
    provider_instance, resolved_model = _configured_provider(provider, model, config)

    results: list[ScenarioResult] = []
    try:
        for mode in modes:
            for scenario in scenarios:
                results.append(_run_scenario(scenario, mode, provider, provider_instance))
    finally:
        close = getattr(provider_instance, "close", None)
        if callable(close):
            close()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"orderflow-eval-{timestamp}"
    if output_root is not None:
        output_base = Path(output_root)
    else:
        configured_output = Path(config.get("output_directory", "evaluation/results"))
        output_base = configured_output if configured_output.is_absolute() else ROOT / configured_output
    run_directory = output_base / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration_file": str(config_path),
        "configuration": config,
        "agent_modes": [mode.value for mode in modes],
        "provider": provider,
        "model": resolved_model,
        "scenario_set": str(scenario_path),
        "random_seed": seed,
        "software": {
            "orderflow_agent_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "catalog_version": JsonCatalogStore().load().version,
            "dependencies": _dependency_versions(
                ("gradio", "nicegui", "openai", "kernelloom", "numpy", "scikit-learn")
            ),
        },
    }
    payload = {"metadata": metadata, "results": [asdict(result) for result in results]}
    (run_directory / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    _write_csv(run_directory / "results.csv", results)
    (run_directory / "summary.md").write_text(_summary(metadata, results), encoding="utf-8")
    (run_directory / "configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (run_directory / "scenario_set.json").write_text(
        json.dumps(json.loads(scenario_path.read_text(encoding="utf-8")), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {"run_directory": str(run_directory), "metadata": metadata, "results": results}


def _configured_provider(provider_id: str, model: str, config: dict[str, Any]) -> tuple[object | None, str]:
    if provider_id == "disabled":
        return None, ""
    settings = RuntimeSettings.from_env()
    settings = replace(
        settings,
        provider_id=provider_id,
        response_model=model or settings.response_model,
        base_url=str(config.get("base_url", settings.base_url)),
        kernelloom_transport=str(
            config.get("kernelloom_transport", settings.kernelloom_transport)
        ),
        kernelloom_chat_model_path=str(
            config.get("kernelloom_chat_model_path", settings.kernelloom_chat_model_path)
        ),
        kernelloom_embedding_model_path=str(
            config.get(
                "kernelloom_embedding_model_path",
                settings.kernelloom_embedding_model_path,
            )
        ),
    )
    return ProviderRegistry.build(settings), settings.response_model


def _run_scenario(
    scenario: Scenario,
    mode: AgentMode,
    provider: str,
    provider_instance: object | None = None,
) -> ScenarioResult:
    with tempfile.TemporaryDirectory(prefix="orderflow-eval-") as directory:
        storage = SQLiteStorageAdapter(Path(directory) / "scenario.db")
        agent = ConversationalTaskAgent(storage=storage, provider=provider_instance)
        session = agent.open_session(mode=mode)
        responses = []
        tool_calls = 0
        latencies = []
        confirmed_ids = []
        triggers = []
        blocked_item_attempts = 0
        for turn in scenario.turns:
            started = time.perf_counter()
            response = agent.handle(turn, session)
            latencies.append((time.perf_counter() - started) * 1000)
            responses.append(response.content)
            tool_calls += len(response.tool_trace)
            blocked_item_attempts += sum(
                step.name == "extract_order" and step.status == "blocked" for step in response.tool_trace
            )
            if response.confirmed_order_id:
                confirmed_ids.append(response.confirmed_order_id)
            if response.handover_decision and response.handover_decision.should_handover:
                triggers.append(response.handover_decision.trigger)
                break

        checks = _expectation_checks(
            scenario.expected,
            session=session,
            orders=storage.list_orders(),
            triggers=triggers,
            responses=responses,
        )
        elapsed = sum(latencies)
        metrics = ScenarioMetrics(
            successful_task_completion=all(checks.values()),
            turns=len(responses),
            tool_calls=tool_calls,
            validation_failures=session.validation_failures,
            corrections_retries=session.repair_requests,
            unsupported_item_attempts=blocked_item_attempts,
            handovers=len(triggers),
            confirmation_failures=session.confirmation_failures,
            latency_ms=round(elapsed, 3),
            mean_turn_latency_ms=round(elapsed / len(latencies), 3) if latencies else 0.0,
            token_usage=None,
        )
        return ScenarioResult(
            scenario_id=scenario.id,
            description=scenario.description,
            agent_mode=mode.value,
            provider=provider,
            passed=all(checks.values()),
            expectation_checks=checks,
            metrics=metrics,
            final_cart=dict(session.order),
            confirmed_order_ids=tuple(confirmed_ids),
            handover_triggers=tuple(triggers),
            assistant_responses=tuple(responses),
        )


def _expectation_checks(
    expected: dict[str, Any],
    *,
    session,
    orders: list[dict[str, Any]],
    triggers: list[str],
    responses: list[str],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if "confirmed_orders" in expected:
        checks["confirmed_orders"] = len(orders) == int(expected["confirmed_orders"])
    if "handover" in expected:
        checks["handover"] = bool(triggers) is bool(expected["handover"])
    if "handover_trigger" in expected:
        checks["handover_trigger"] = expected["handover_trigger"] in triggers
    if "final_cart" in expected:
        checks["final_cart"] = dict(session.order) == dict(expected["final_cart"])
    if "confirmed_total" in expected:
        checks["confirmed_total"] = bool(orders) and orders[-1]["total"] == int(expected["confirmed_total"])
    if "response_contains" in expected:
        transcript = "\n".join(responses)
        checks["response_contains"] = all(str(value) in transcript for value in expected["response_contains"])
    if "pending_action" in expected:
        checks["pending_action"] = session.pending_action == str(expected["pending_action"])
    if "confirmation_failures" in expected:
        checks["confirmation_failures"] = session.confirmation_failures == int(
            expected["confirmation_failures"]
        )
    if "validation_failures" in expected:
        checks["validation_failures"] = session.validation_failures == int(
            expected["validation_failures"]
        )
    if "unsupported_item_attempts" in expected:
        checks["unsupported_item_attempts"] = session.unsupported_attempts == int(
            expected["unsupported_item_attempts"]
        )
    if "handover_active" in expected:
        checks["handover_active"] = session.handover_active is bool(expected["handover_active"])
    return checks or {"scenario_executed": True}


def _write_csv(path: Path, results: list[ScenarioResult]) -> None:
    fields = (
        "scenario_id",
        "description",
        "agent_mode",
        "provider",
        "passed",
        "successful_task_completion",
        "turns",
        "tool_calls",
        "validation_failures",
        "corrections_retries",
        "unsupported_item_attempts",
        "handovers",
        "confirmation_failures",
        "latency_ms",
        "mean_turn_latency_ms",
        "token_usage",
        "handover_triggers",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scenario_id": result.scenario_id,
                    "description": result.description,
                    "agent_mode": result.agent_mode,
                    "provider": result.provider,
                    "passed": result.passed,
                    **asdict(result.metrics),
                    "handover_triggers": ",".join(result.handover_triggers),
                }
            )


def _summary(metadata: dict[str, Any], results: list[ScenarioResult]) -> str:
    passed = sum(result.passed for result in results)
    lines = [
        "# OrderFlow-Agent scenario evaluation",
        "",
        f"Run: `{metadata['run_id']}`",
        f"Provider: `{metadata['provider']}`",
        f"Scenarios: {len(results)} mode/scenario combinations",
        f"Expectation checks passed: {passed}/{len(results)}",
        "",
        "This automated replay measures software behaviour. It does not measure satisfaction, trust, or causal effects.",
        "",
        "| Mode | Scenario | Passed | Turns | Tools | Handover |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {result.agent_mode} | {result.scenario_id} | {'yes' if result.passed else 'no'} | "
        f"{result.metrics.turns} | {result.metrics.tool_calls} | {result.metrics.handovers} |"
        for result in results
    )
    return "\n".join(lines) + "\n"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeatable OrderFlow pizza-order scenarios.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_evaluation(args.config, output_root=args.output)
    print(result["run_directory"])
    return 0 if all(item.passed for item in result["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
