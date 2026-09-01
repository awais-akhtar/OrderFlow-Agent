from __future__ import annotations

import unittest

from orderflow_agent.runtime.analysis import condition_summary


class ConditionAnalysisTest(unittest.TestCase):
    def test_summarises_verified_ordering_metrics_by_agent_mode(self) -> None:
        rows = [
            {
                "agent_mode": "controlled",
                "strictness": 80,
                "confirmed_orders": 1,
                "repair_requests": 2,
                "successful_repairs": 1,
                "compliance_failures": 0,
            },
            {
                "agent_mode": "controlled",
                "strictness": 80,
                "confirmed_orders": 0,
                "repair_requests": 0,
                "successful_repairs": 0,
                "compliance_failures": 1,
            },
        ]

        summary = condition_summary(rows)[0]

        self.assertEqual(summary["agent_mode"], "controlled")
        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(summary["verified_task_success_rate"], 0.5)
        self.assertEqual(summary["repair_success_rate"], 0.5)
        self.assertEqual(summary["compliance_failures"], 1)


if __name__ == "__main__":
    unittest.main()
