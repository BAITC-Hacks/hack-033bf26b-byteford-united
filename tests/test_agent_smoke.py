import unittest

from agent import Agent
from mock_environment import make_mock_env


class AgentSmokeTest(unittest.TestCase):
    def test_agent_respects_environment_limits(self):
        env, _ = make_mock_env(seed=7)

        campaigns = Agent().act(env)

        self.assertGreaterEqual(len(campaigns), 1)
        self.assertLessEqual(len(campaigns), 10)
        self.assertGreaterEqual(env.remaining_budget, 0)
        self.assertGreaterEqual(env.remaining_contacts, 0)
        self.assertGreaterEqual(env.pilots_left, 0)
        for campaign in campaigns:
            self.assertIn(campaign["target_tariff"], set(env.tariffs["tariff_plan_code"]))
            self.assertIn(campaign["channel"], env.channels)


if __name__ == "__main__":
    unittest.main()
