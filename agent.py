"""Hackathon agent entry point.

The orchestration and experiment policy live in ``strategy.py``.  A teammate
can add ``priors.py`` independently; if it is absent or fails, the agent keeps
working with the deterministic tariff-fit fallback.
"""

from pathlib import Path

from strategy import CampaignStrategy


class Agent:
    """Select campaigns after a bounded sequence of adaptive pilots."""

    def act(self, env) -> list[dict]:
        strategy = CampaignStrategy(base_dir=Path(__file__).resolve().parent)
        return strategy.run(env)
