"""Compare risk-policy configurations on identical stress scenarios."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from agent import Agent
from stress_benchmark import SCENARIOS, evaluate_stress_run
from strategy import CampaignStrategy, StrategyConfig


@dataclass(frozen=True)
class ConfiguredAgent:
    config: StrategyConfig

    def act(self, env) -> list[dict]:
        from pathlib import Path

        return CampaignStrategy(
            base_dir=Path(__file__).resolve().parent,
            config=self.config,
        ).run(env)


PROFILES = {
    "legacy_14_6": StrategyConfig(
        initial_pilots=14,
        followup_pilots=6,
        initial_observed_quota=12,
        initial_target_fallback_quota=1,
        initial_price_fallback_quota=1,
        min_pilots_for_rollout=1,
    ),
    "current_10_10": StrategyConfig(),
    "weaker_priors": StrategyConfig(
        initial_pilots=14,
        followup_pilots=6,
        initial_observed_quota=12,
        initial_target_fallback_quota=1,
        initial_price_fallback_quota=1,
        min_prior_std=0.30,
        min_pilots_for_rollout=1,
    ),
    "more_confirmation": StrategyConfig(
        initial_pilots=12,
        followup_pilots=8,
        initial_observed_quota=10,
        initial_target_fallback_quota=1,
        initial_price_fallback_quota=1,
        min_pilots_for_rollout=1,
    ),
    "higher_safety": StrategyConfig(
        initial_pilots=14,
        followup_pilots=6,
        initial_observed_quota=12,
        initial_target_fallback_quota=1,
        initial_price_fallback_quota=1,
        lcb_z=1.96,
        min_pilots_for_rollout=1,
    ),
    "robust_combo": StrategyConfig(
        initial_pilots=12,
        followup_pilots=8,
        initial_observed_quota=10,
        initial_target_fallback_quota=1,
        initial_price_fallback_quota=1,
        min_prior_std=0.30,
        lcb_z=1.96,
        min_pilots_for_rollout=1,
    ),
    "confirmed_only": StrategyConfig(min_pilots_for_rollout=2),
    "balanced_confirmed": StrategyConfig(
        initial_pilots=12,
        followup_pilots=8,
        initial_observed_quota=10,
        initial_target_fallback_quota=1,
        initial_price_fallback_quota=1,
        min_pilots_for_rollout=2,
    ),
}


def run_comparison(
    *,
    runs: int,
    start_seed: int = 0,
    profiles: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows = []
    selected_profiles = profiles or tuple(PROFILES)
    unknown = sorted(set(selected_profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    for profile_name in selected_profiles:
        config = PROFILES[profile_name]
        factory = (
            Agent
            if profile_name == "current_10_10"
            else lambda selected=config: ConfiguredAgent(selected)
        )
        for scenario_index, scenario in enumerate(SCENARIOS):
            for seed in range(start_seed, start_seed + runs):
                row = evaluate_stress_run(
                    scenario=scenario,
                    run_seed=seed,
                    model_seed=100_000 + scenario_index * 10_000 + seed,
                    agent_factory=factory,
                )
                row["profile"] = profile_name
                rows.append(row)
    return pd.DataFrame(rows)


def _risk_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, group in frame.groupby("profile", sort=False):
        shifted = group[group["scenario"] != "reference"]
        reference = group[group["scenario"] == "reference"]
        shifted_net = shifted["net_gain"]
        cutoff = float(shifted_net.quantile(0.20))
        rows.append(
            {
                "profile": profile,
                "reference_median": float(reference["net_gain"].median()),
                "shifted_positive": int((shifted_net > 0).sum()),
                "shifted_runs": len(shifted),
                "shifted_median": float(shifted_net.median()),
                "shifted_p10": float(shifted_net.quantile(0.10)),
                "shifted_worst20_mean": float(
                    shifted_net[shifted_net <= cutoff].mean()
                ),
                "shifted_minimum": float(shifted_net.min()),
                "average_cost": float(shifted["cost"].mean()),
                "average_risk": float(shifted["risk_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["shifted_worst20_mean", "shifted_p10"], ascending=False
    )


def _scenario_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (profile, scenario), group in frame.groupby(
        ["profile", "scenario"], sort=False
    ):
        net = group["net_gain"]
        rows.append(
            {
                "profile": profile,
                "scenario": scenario,
                "positive": f"{int((net > 0).sum())}/{len(net)}",
                "median": float(net.median()),
                "p10": float(net.quantile(0.10)),
                "minimum": float(net.min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exploration and risk configurations"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILES),
        default=None,
        help="optional subset of named profiles",
    )
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")

    frame = run_comparison(
        runs=args.runs,
        start_seed=args.start_seed,
        profiles=tuple(args.profiles) if args.profiles else None,
    )
    summary = _risk_summary(frame)
    numeric_columns = [
        "reference_median",
        "shifted_median",
        "shifted_p10",
        "shifted_worst20_mean",
        "shifted_minimum",
        "average_cost",
    ]
    print(
        summary.to_string(
            index=False,
            formatters={
                column: (lambda value: f"{value:,.0f}")
                for column in numeric_columns
            },
        )
    )
    print(
        "\nProfiles are ranked by the mean of the worst 20% shifted runs. "
        "Keep the default unchanged until a candidate wins on more seeds."
    )
    if len(frame["profile"].unique()) <= 2:
        scenario_summary = _scenario_summary(frame)
        print("\nPer-scenario validation")
        print(
            scenario_summary.to_string(
                index=False,
                formatters={
                    column: (lambda value: f"{value:,.0f}")
                    for column in ["median", "p10", "minimum"]
                },
            )
        )


if __name__ == "__main__":
    main()
