import unittest

from agent import Agent
from mock_environment import make_mock_env
from scoring_core import apply_filters
from strategy import CampaignStrategy


class AgentSmokeTest(unittest.TestCase):
    def test_agent_respects_environment_limits(self):
        env, _ = make_mock_env(seed=7)

        campaigns = Agent().act(env)

        self.assertGreaterEqual(len(campaigns), 1)
        self.assertLessEqual(len(campaigns), 10)
        self.assertGreaterEqual(env.remaining_budget, 0)
        self.assertGreaterEqual(env.remaining_contacts, 0)
        self.assertGreaterEqual(env.pilots_left, 0)
        final_contacts = 0
        final_cost = 0.0
        for campaign in campaigns:
            self.assertIn(campaign["target_tariff"], set(env.tariffs["tariff_plan_code"]))
            self.assertIn(campaign["channel"], env.channels)
            segment = apply_filters(env.customer_profile, campaign).head(5_000)
            final_contacts += len(segment)
            final_cost += len(segment) * env.channels[campaign["channel"]]["cost_per_contact"]
        self.assertLessEqual(final_contacts, env.remaining_contacts)
        self.assertLessEqual(final_cost, env.remaining_budget)

    def test_fallback_strategy_works_without_external_priors(self):
        env, _ = make_mock_env(seed=11)

        campaigns = CampaignStrategy(
            base_dir=".", use_external_priors=False
        ).run(env)

        self.assertGreaterEqual(len(campaigns), 1)
        self.assertLessEqual(len(campaigns), 10)

    def test_initial_exploration_includes_unobserved_transitions(self):
        env, _ = make_mock_env(seed=13)
        strategy = CampaignStrategy(base_dir=".")
        candidates = strategy._load_candidates(env)

        strategy._run_initial_pilots(env, candidates)

        tested_sources = {
            candidate.source for candidate in candidates if candidate.pilot_count > 0
        }
        self.assertIn("observed_smoothed", tested_sources)
        self.assertTrue(
            tested_sources
            & {"target_history_fallback", "tariff_price_fallback"}
        )


if __name__ == "__main__":
    unittest.main()
