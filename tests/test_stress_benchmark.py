import unittest

import pandas as pd

from mock_environment import _mock_impact_model
from stress_benchmark import (
    SCENARIOS,
    build_scenario_model,
    evaluate_stress_run,
)


class StressBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        history = pd.read_csv("data/change_tariff.csv")
        cls.base_model = _mock_impact_model(history)

    def test_scenarios_are_deterministic_and_do_not_mutate_base(self):
        original = self.base_model.copy(deep=True)
        for scenario in SCENARIOS:
            first = build_scenario_model(self.base_model, scenario, seed=123)
            second = build_scenario_model(self.base_model, scenario, seed=123)
            pd.testing.assert_frame_equal(first, second)
            self.assertTrue(first["arpu_change_pct"].between(-1.0, 3.0).all())
            self.assertTrue(first["conversion_rate"].between(0.001, 1.0).all())
        pd.testing.assert_frame_equal(self.base_model, original)

    def test_reference_run_obeys_agent_contract(self):
        result = evaluate_stress_run(
            scenario="reference",
            run_seed=42,
            model_seed=100_000,
        )
        self.assertGreater(result["pilots"], 0)
        self.assertLessEqual(result["pilots"], 20)
        self.assertGreaterEqual(result["final_campaigns"], 1)
        self.assertLessEqual(result["final_campaigns"], 10)
        self.assertLessEqual(result["contacts"], 15_000)
        self.assertLessEqual(result["cost"], 100_000)


if __name__ == "__main__":
    unittest.main()
