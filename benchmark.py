"""Compare the complete agent against its tariff-fit fallback.

This is a development tool only.  It uses the public mock environment through
the same ``evaluate_agent`` function as ``local_eval.py`` and never accesses
hidden environment state.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from local_eval import evaluate_agent
from strategy import CampaignStrategy


BASE_DIR = Path(__file__).resolve().parent


class StrategyVariantAgent:
    def __init__(self, *, use_priors: bool):
        self.use_priors = use_priors
        self.final_campaign_count = 0

    def act(self, env) -> list[dict]:
        campaigns = CampaignStrategy(
            base_dir=BASE_DIR,
            use_external_priors=self.use_priors,
        ).run(env)
        self.final_campaign_count = len(campaigns)
        return campaigns


def _run_variant(*, use_priors: bool, seeds: range) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in seeds:
        agent = StrategyVariantAgent(use_priors=use_priors)
        result = evaluate_agent(agent, seed=seed, verbose=False)
        if result is None:
            raise RuntimeError(f"Evaluation returned no result for seed {seed}")
        rows.append(
            {
                "seed": seed,
                "net_gain": float(result["net_arpu_gain"]),
                "gross_lift": float(result["gross_arpu_lift"]),
                "cost": float(result["total_cost"]),
                "contacts": int(result["total_contacts"]),
                "pilots": int(result["n_pilots"]),
                "final_campaigns": int(agent.final_campaign_count),
            }
        )
    return pd.DataFrame(rows).set_index("seed")


def _summary(frame: pd.DataFrame) -> dict[str, float | int]:
    net = frame["net_gain"]
    return {
        "median_net": float(net.median()),
        "p10_net": float(net.quantile(0.10)),
        "min_net": float(net.min()),
        "max_net": float(net.max()),
        "positive_runs": int((net > 0).sum()),
        "avg_cost": float(frame["cost"].mean()),
        "avg_contacts": float(frame["contacts"].mean()),
        "avg_final_campaigns": float(frame["final_campaigns"].mean()),
    }


def _print_summary(name: str, frame: pd.DataFrame) -> None:
    summary = _summary(frame)
    print(f"\n{name}")
    print(f"  median net:          {summary['median_net']:>14,.0f}")
    print(f"  p10 net:             {summary['p10_net']:>14,.0f}")
    print(f"  minimum net:         {summary['min_net']:>14,.0f}")
    print(f"  maximum net:         {summary['max_net']:>14,.0f}")
    print(f"  positive runs:       {summary['positive_runs']:>14} / {len(frame)}")
    print(f"  average cost:        {summary['avg_cost']:>14,.0f}")
    print(f"  average contacts:    {summary['avg_contacts']:>14,.0f}")
    print(f"  avg final campaigns: {summary['avg_final_campaigns']:>14.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B benchmark: tariff-fit fallback versus historical priors"
    )
    parser.add_argument("--runs", type=int, default=10, help="number of seeds")
    parser.add_argument("--start-seed", type=int, default=0, help="first seed")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")

    # local_eval.py loads the participant files by relative path.  Changing the
    # process working directory keeps this command usable from any folder.
    os.chdir(BASE_DIR)
    seeds = range(args.start_seed, args.start_seed + args.runs)

    fallback = _run_variant(use_priors=False, seeds=seeds)
    with_priors = _run_variant(use_priors=True, seeds=seeds)

    comparison = pd.DataFrame(
        {
            "fallback_net": fallback["net_gain"],
            "priors_net": with_priors["net_gain"],
        }
    )
    comparison["delta"] = comparison["priors_net"] - comparison["fallback_net"]

    print("seed comparison")
    print(comparison.to_string(float_format=lambda value: f"{value:,.0f}"))
    _print_summary("tariff-fit fallback", fallback)
    _print_summary("historical priors", with_priors)

    median_delta = float(comparison["delta"].median())
    print(f"\nmedian priors uplift over fallback: {median_delta:,.0f}")
    print(
        "Note: the mock uses the supplied history to generate effects. "
        "This benchmark validates integration and stability, not the hidden judging score."
    )


if __name__ == "__main__":
    main()
