from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from orderflow_agent.evaluation.runner import run_evaluation
from orderflow_agent.modes import AgentMode, coerce_mode


ROOT = Path(__file__).resolve().parent.parent


class AgentModeTest(unittest.TestCase):
    def test_mode_configuration_and_legacy_aliases(self) -> None:
        self.assertEqual(coerce_mode("controlled"), AgentMode.CONTROLLED)
        self.assertEqual(coerce_mode("guided"), AgentMode.ASSISTED)
        self.assertEqual(coerce_mode("adaptive"), AgentMode.FLEXIBLE)
        self.assertEqual(AgentMode.CONTROLLED.strictness, 80)
        self.assertEqual(AgentMode.ASSISTED.strictness, 50)
        self.assertEqual(AgentMode.FLEXIBLE.strictness, 20)


class EvaluationRunnerTest(unittest.TestCase):
    def test_runner_replays_every_scenario_in_every_mode_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_evaluation(
                ROOT / "evaluation" / "configs" / "default.json",
                output_root=Path(directory),
            )
            results = result["results"]
            self.assertEqual(len(results), 54)
            self.assertTrue(all(row.passed for row in results))
            self.assertEqual({row.agent_mode for row in results}, {mode.value for mode in AgentMode})

            output = Path(result["run_directory"])
            self.assertTrue((output / "configuration.json").exists())
            self.assertTrue((output / "results.json").exists())
            self.assertTrue((output / "results.csv").exists())
            self.assertTrue((output / "summary.md").exists())
            self.assertTrue((output / "scenario_set.json").exists())

            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["random_seed"], 42)
            self.assertIn("orderflow_agent_version", payload["metadata"]["software"])
            self.assertIn("catalog_version", payload["metadata"]["software"])
            self.assertEqual(payload["metadata"]["software"]["dependencies"]["kernelloom"], "0.4.1")
            self.assertIsNone(payload["results"][0]["metrics"]["token_usage"])

            with (output / "results.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 54)
            self.assertIn("tool_calls", rows[0])
            self.assertIn("validation_failures", rows[0])
            self.assertIn("confirmation_failures", rows[0])

            summary = (output / "summary.md").read_text(encoding="utf-8")
            self.assertIn("does not measure satisfaction", summary)


if __name__ == "__main__":
    unittest.main()
