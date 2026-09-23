"""Evaluate the agent when live effects differ from the historical mock.

The regular mock is intentionally derived from ``change_tariff.csv`` and is
therefore favourable to history-based priors.  This development benchmark
perturbs the *environment* impact model while the agent still sees the same
participant data.  It measures robustness to distribution shift without
exposing the shifted model through ``env``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from agent import Agent
from environment import make_environment
from mock_environment import (
    CHANNELS,
    MAX_TOTAL_CONTACTS,
    TOTAL_BUDGET,
    _mock_fallback,
    _mock_impact_model,
)
from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns


BASE_DIR = Path(__file__).resolve().parent
SCENARIOS = (
    "reference",
    "effect_noise",
    "conversion_shift",
    "partial_reversal",
    "combined_shift",
)


@dataclass(frozen=True)
class StressConfig:
    effect_noise_std: float = 0.18
    conversion_log_std: float = 0.35
    reversal_share: float = 0.20
    reversal_scale: float = 0.75


def build_scenario_model(
    base_model: pd.DataFrame,
    scenario: str,
    *,
    seed: int,
    config: StressConfig = StressConfig(),
) -> pd.DataFrame:
    """Return a deterministic shifted copy of the public mock impact model."""

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    model = base_model.copy(deep=True)
    if scenario == "reference":
        return model

    rng = np.random.default_rng(seed)
    apply_effect_noise = scenario in {"effect_noise", "combined_shift"}
    apply_conversion_shift = scenario in {"conversion_shift", "combined_shift"}
    apply_reversal = scenario in {"partial_reversal", "combined_shift"}

    if apply_effect_noise:
        model["arpu_change_pct"] = (
            model["arpu_change_pct"]
            + rng.normal(0.0, config.effect_noise_std, size=len(model))
        ).clip(-1.0, 3.0)

    if apply_conversion_shift:
        model["conversion_rate"] = (
            model["conversion_rate"]
            * np.exp(rng.normal(0.0, config.conversion_log_std, size=len(model)))
        ).clip(0.001, 1.0)

    if apply_reversal:
        n_reversed = max(1, round(len(model) * config.reversal_share))
        reversed_indices = rng.choice(model.index.to_numpy(), size=n_reversed, replace=False)
        model.loc[reversed_indices, "arpu_change_pct"] = (
            -config.reversal_scale
            * model.loc[reversed_indices, "arpu_change_pct"]
        ).clip(-1.0, 3.0)

    return model


def evaluate_stress_run(
    *,
    scenario: str,
    run_seed: int,
    model_seed: int,
    config: StressConfig = StressConfig(),
    agent_factory: Callable[[], object] = Agent,
) -> dict:
    """Run the real Agent contract against one shifted hidden environment."""

    profile = pd.read_csv(BASE_DIR / "customer_profile.csv")
    tariffs = pd.read_csv(BASE_DIR / "data" / "dict_tariff.csv")
    history = pd.read_csv(BASE_DIR / "data" / "change_tariff.csv")
    base_model = _mock_impact_model(history)
    impact_model = build_scenario_model(
        base_model,
        scenario,
        seed=model_seed,
        config=config,
    )

    env, internals = make_environment(
        customer_profile=profile,
        impact_model=impact_model,
        dict_tariff=tariffs,
        channels=CHANNELS,
        total_budget=TOTAL_BUDGET,
        max_total_contacts=MAX_TOTAL_CONTACTS,
        fallback_predict=_mock_fallback,
        seed=run_seed,
    )
    agent = agent_factory()
    final_campaigns = sanitize_campaigns(agent.act(env), env.tariffs)[:MAX_CAMPAIGNS]
    pilots = internals.executed_pilot_campaigns()
    campaigns = pd.DataFrame(pilots + final_campaigns)
    for column in [
        "filter_arpu_segment",
        "filter_data_segment",
        "filter_call_segment",
        "filter_current_tariff",
        "explicit_ids",
    ]:
        if column not in campaigns.columns:
            campaigns[column] = None

    result = score_campaigns(
        campaigns,
        profile,
        impact_model,
        tariffs,
        float(profile["predicted_arpu"].sum()),
        _mock_fallback,
        team_id=f"stress-{scenario}",
    )
    return {
        "scenario": scenario,
        "run_seed": run_seed,
        "model_seed": model_seed,
        "net_gain": float(result["net_arpu_gain"]),
        "gross_lift": float(result["gross_arpu_lift"]),
        "cost": float(result["total_cost"]),
        "contacts": int(result["total_contacts"]),
        "unique_customers": int(result["unique_customers_targeted"]),
        "risk_score": float(result["risk_score_pct"]),
        "pilots": len(pilots),
        "final_campaigns": len(final_campaigns),
    }


def run_stress_benchmark(
    *,
    runs: int,
    start_seed: int = 0,
    agent_factory: Callable[[], object] = Agent,
) -> pd.DataFrame:
    if runs <= 0:
        raise ValueError("runs must be positive")
    rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in range(start_seed, start_seed + runs):
            rows.append(
                evaluate_stress_run(
                    scenario=scenario,
                    run_seed=seed,
                    model_seed=100_000 + scenario_index * 10_000 + seed,
                    agent_factory=agent_factory,
                )
            )
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Return risk-focused metrics rather than only the best mock score."""

    rows = []
    for scenario, group in frame.groupby("scenario", sort=False):
        net = group["net_gain"]
        cutoff = float(net.quantile(0.20))
        downside = net[net <= cutoff]
        rows.append(
            {
                "scenario": scenario,
                "positive_runs": int((net > 0).sum()),
                "runs": len(group),
                "median_net": float(net.median()),
                "p10_net": float(net.quantile(0.10)),
                "worst_20pct_mean": float(downside.mean()),
                "minimum_net": float(net.min()),
                "average_cost": float(group["cost"].mean()),
                "average_contacts": float(group["contacts"].mean()),
                "average_risk_score": float(group["risk_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress benchmark under shifted hidden campaign effects"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")

    frame = run_stress_benchmark(runs=args.runs, start_seed=args.start_seed)
    summary = summarise(frame)
    print(
        summary.to_string(
            index=False,
            formatters={
                column: (lambda value: f"{value:,.0f}")
                for column in [
                    "median_net",
                    "p10_net",
                    "worst_20pct_mean",
                    "minimum_net",
                    "average_cost",
                    "average_contacts",
                ]
            },
        )
    )
    print(
        "\nReference reuses the normal mock. Shifted scenarios are development "
        "stress tests, not predictions of the private judging model."
    )
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = BASE_DIR / output
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"Detailed runs saved to: {output}")


if __name__ == "__main__":
    main()
