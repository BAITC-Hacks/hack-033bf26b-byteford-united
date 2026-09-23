"""Historical priors and campaign candidates for the tariff agent.

The judging population is deliberately different from the historical sample.
Consequently, this module does not treat historical averages as truth.  It
builds weak, smoothed priors that are useful for deciding which transitions to
pilot first.  Pilot observations should later update or replace these values.

Public entry points
-------------------
``build_transition_priors``
    One row per (current tariff, ARPU segment, target tariff), including rows
    that were never observed in the history.

``build_candidate_table``
    Joins the priors to the current audience and expands every transition over
    the available communication channels.

``select_pilot_candidates``
    Produces a small, diverse shortlist for the exploration stage.

The implementation intentionally uses only pandas, numpy and the Python
standard library.  It never reads environment variables or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ARPU_LABELS = ("LOW", "MID", "HIGH")
ARPU_BINS = (-np.inf, 1_000.0, 5_000.0, np.inf)

DEFAULT_CHANNELS = {
    "push": {"cost_per_contact": 0.0, "conversion_multiplier": 0.50},
    "sms": {"cost_per_contact": 4.0, "conversion_multiplier": 0.65},
    "digital_ads": {"cost_per_contact": 22.0, "conversion_multiplier": 0.85},
    "call": {"cost_per_contact": 160.0, "conversion_multiplier": 1.20},
}


@dataclass(frozen=True)
class PriorConfig:
    """Tunable assumptions for empirical-Bayes smoothing.

    ``lift_prior_strength`` and ``conversion_prior_strength`` are equivalent
    sample sizes.  Larger values make sparse historical groups fall back more
    strongly to broad historical and tariff-price information.
    """

    min_previous_arpu: float = 100.0
    change_clip_low: float = -1.0
    change_clip_high: float = 3.0
    lift_prior_strength: float = 30.0
    conversion_prior_strength: float = 20.0
    fallback_price_weight: float = 0.40
    conservative_z: float = 1.0
    max_customers_per_campaign: int = 5_000


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _clean_tariffs(tariffs: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        tariffs,
        {"tariff_plan_code", "price_tariff"},
        "tariffs",
    )
    result = tariffs.copy()
    result["tariff_plan_code"] = result["tariff_plan_code"].astype(str)
    result["price_tariff"] = pd.to_numeric(result["price_tariff"], errors="coerce")
    result = result.dropna(subset=["tariff_plan_code"]).drop_duplicates(
        "tariff_plan_code", keep="last"
    )
    if result.empty:
        raise ValueError("tariffs contains no usable tariff_plan_code values")
    return result


def _price_fallback_series(
    current_tariff: pd.Series,
    target_tariff: pd.Series,
    tariffs: pd.DataFrame,
    config: PriorConfig,
) -> pd.Series:
    """Conservative fallback lift based on the monthly tariff price gap."""

    prices = tariffs.set_index("tariff_plan_code")["price_tariff"]
    scale = float(prices[prices > 0].median()) if (prices > 0).any() else 1.0
    current_price = current_tariff.map(prices)
    target_price = target_tariff.map(prices)
    signal = config.fallback_price_weight * (target_price - current_price) / max(scale, 1.0)
    return signal.fillna(0.0).clip(config.change_clip_low, config.change_clip_high)


def fallback_transition_prior(
    current_tariff: str,
    target_tariff: str,
    tariffs: pd.DataFrame,
    *,
    conversion_rate: float | None = None,
    config: PriorConfig = PriorConfig(),
) -> dict[str, float | str | bool]:
    """Return a standalone fallback for a transition absent from history.

    This function is useful when the live profile contains a tariff that was
    not present while the complete prior table was built.  The estimate is
    deliberately weak and marked as low confidence.
    """

    clean_tariffs = _clean_tariffs(tariffs)
    current = pd.Series([str(current_tariff)])
    target = pd.Series([str(target_tariff)])
    change = float(_price_fallback_series(current, target, clean_tariffs, config).iloc[0])
    if conversion_rate is None:
        conversion_rate = 1.0 / max(len(clean_tariffs) - 1, 1)
    conversion = float(np.clip(conversion_rate, 0.0, 1.0))
    return {
        "smoothed_change_pct": change,
        "smoothed_conversion_rate": conversion,
        "expected_base_lift_ratio": change * conversion,
        "history_confidence": 0.0,
        "observed_in_history": False,
        "prior_source": "tariff_price_fallback",
    }


def build_transition_priors(
    change_tariff: pd.DataFrame,
    tariffs: pd.DataFrame,
    *,
    config: PriorConfig = PriorConfig(),
) -> pd.DataFrame:
    """Build a complete, smoothed table of transition priors.

    Relative ARPU changes are clipped to the same public range used by the mock
    environment.  The lift is shrunk towards a blend of target-level history
    and a conservative tariff-price signal.  Transition frequency is smoothed
    with a Dirichlet-style prior based on target popularity inside each ARPU
    segment.
    """

    _require_columns(
        change_tariff,
        {
            "AVG_ARPU_PREV_3M",
            "AVG_ARPU_NEXT_3M",
            "tariff_plan_code_from",
            "tariff_plan_code_to",
        },
        "change_tariff",
    )
    clean_tariffs = _clean_tariffs(tariffs)
    tariff_codes = clean_tariffs["tariff_plan_code"].tolist()
    known_tariffs = set(tariff_codes)

    history = change_tariff.copy()
    history["AVG_ARPU_PREV_3M"] = pd.to_numeric(
        history["AVG_ARPU_PREV_3M"], errors="coerce"
    )
    history["AVG_ARPU_NEXT_3M"] = pd.to_numeric(
        history["AVG_ARPU_NEXT_3M"], errors="coerce"
    )
    history = history.dropna(
        subset=[
            "AVG_ARPU_PREV_3M",
            "AVG_ARPU_NEXT_3M",
            "tariff_plan_code_from",
            "tariff_plan_code_to",
        ]
    )
    history["tariff_plan_code_from"] = history["tariff_plan_code_from"].astype(str)
    history["tariff_plan_code_to"] = history["tariff_plan_code_to"].astype(str)
    history = history[
        history["tariff_plan_code_from"].isin(known_tariffs)
        & history["tariff_plan_code_to"].isin(known_tariffs)
        & (history["tariff_plan_code_from"] != history["tariff_plan_code_to"])
        & (history["AVG_ARPU_PREV_3M"] >= config.min_previous_arpu)
    ].copy()

    if history.empty:
        raise ValueError("change_tariff contains no usable transition rows")

    history["arpu_segment"] = pd.cut(
        history["AVG_ARPU_PREV_3M"],
        bins=ARPU_BINS,
        labels=ARPU_LABELS,
    ).astype("object")
    history["arpu_change_pct"] = (
        (history["AVG_ARPU_NEXT_3M"] - history["AVG_ARPU_PREV_3M"])
        / history["AVG_ARPU_PREV_3M"]
    ).clip(config.change_clip_low, config.change_clip_high)

    keys = ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"]
    pair = (
        history.groupby(keys, observed=True)["arpu_change_pct"]
        .agg(
            n_observations="size",
            raw_change_mean="mean",
            raw_change_median="median",
            raw_change_std="std",
        )
        .reset_index()
    )
    direction = history.assign(
        is_up=(history["arpu_change_pct"] > 0.10).astype(float),
        is_down=(history["arpu_change_pct"] < -0.10).astype(float),
    )
    direction = (
        direction.groupby(keys, observed=True)
        .agg(upsell_rate=("is_up", "mean"), downsell_rate=("is_down", "mean"))
        .reset_index()
    )
    pair = pair.merge(direction, on=keys, how="left")

    from_totals = (
        history.groupby(["tariff_plan_code_from", "arpu_segment"], observed=True)
        .size()
        .rename("transition_total")
        .reset_index()
    )
    target_stats = (
        history.groupby(["arpu_segment", "tariff_plan_code_to"], observed=True)
        .agg(
            target_segment_n=("arpu_change_pct", "size"),
            target_segment_mean=("arpu_change_pct", "mean"),
        )
        .reset_index()
    )
    segment_stats = (
        history.groupby("arpu_segment", observed=True)["arpu_change_pct"]
        .agg(segment_n="size", segment_mean="mean", segment_std="std")
        .reset_index()
    )

    target_counts = (
        history.groupby(["arpu_segment", "tariff_plan_code_to"], observed=True)
        .size()
        .rename("target_count")
        .reset_index()
    )
    segment_counts = (
        history.groupby("arpu_segment", observed=True)
        .size()
        .rename("segment_transition_count")
        .reset_index()
    )

    grid = pd.MultiIndex.from_product(
        [tariff_codes, ARPU_LABELS, tariff_codes],
        names=keys,
    ).to_frame(index=False)
    grid = grid[grid["tariff_plan_code_from"] != grid["tariff_plan_code_to"]].copy()

    result = grid.merge(pair, on=keys, how="left")
    result = result.merge(
        from_totals,
        on=["tariff_plan_code_from", "arpu_segment"],
        how="left",
    )
    result = result.merge(
        target_stats,
        on=["arpu_segment", "tariff_plan_code_to"],
        how="left",
    )
    result = result.merge(segment_stats, on="arpu_segment", how="left")
    result = result.merge(
        target_counts,
        on=["arpu_segment", "tariff_plan_code_to"],
        how="left",
    )
    result = result.merge(segment_counts, on="arpu_segment", how="left")

    count_columns = [
        "n_observations",
        "transition_total",
        "target_segment_n",
        "target_count",
        "segment_transition_count",
    ]
    result[count_columns] = result[count_columns].fillna(0)

    # Add-one target probabilities ensure that never-observed targets still get
    # a small but non-zero conversion prior.
    result["target_prior_probability"] = (
        result["target_count"] + 1.0
    ) / (result["segment_transition_count"] + len(tariff_codes))

    price_signal = _price_fallback_series(
        result["tariff_plan_code_from"],
        result["tariff_plan_code_to"],
        clean_tariffs,
        config,
    )
    broad_mean = result["target_segment_mean"].fillna(result["segment_mean"]).fillna(0.0)
    broad_weight = result["target_segment_n"] / (
        result["target_segment_n"] + config.lift_prior_strength
    )
    result["prior_center_change_pct"] = (
        broad_weight * broad_mean + (1.0 - broad_weight) * price_signal
    ).clip(config.change_clip_low, config.change_clip_high)

    n = result["n_observations"]
    result["smoothed_change_pct"] = (
        n * result["raw_change_mean"].fillna(0.0)
        + config.lift_prior_strength * result["prior_center_change_pct"]
    ) / (n + config.lift_prior_strength)

    result["raw_conversion_rate"] = np.where(
        result["transition_total"] > 0,
        n / result["transition_total"],
        0.0,
    )
    result["smoothed_conversion_rate"] = (
        n + config.conversion_prior_strength * result["target_prior_probability"]
    ) / (result["transition_total"] + config.conversion_prior_strength)
    result["smoothed_conversion_rate"] = result["smoothed_conversion_rate"].clip(0.0, 1.0)

    global_std = float(history["arpu_change_pct"].std(ddof=1))
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = 0.804
    fallback_std = result["segment_std"].fillna(global_std).clip(lower=1e-6)
    observed_std = result["raw_change_std"].fillna(fallback_std).clip(lower=1e-6)
    pooled_variance = (
        np.maximum(n - 1.0, 0.0) * observed_std.pow(2)
        + config.lift_prior_strength * fallback_std.pow(2)
    ) / (np.maximum(n - 1.0, 0.0) + config.lift_prior_strength)
    result["prior_standard_error"] = np.sqrt(
        pooled_variance / (n + config.lift_prior_strength)
    )
    result["history_confidence"] = n / (n + config.lift_prior_strength)
    result["observed_in_history"] = n > 0
    result["prior_source"] = np.select(
        [n > 0, result["target_segment_n"] > 0],
        ["observed_smoothed", "target_history_fallback"],
        default="tariff_price_fallback",
    )
    result["expected_base_lift_ratio"] = (
        result["smoothed_change_pct"] * result["smoothed_conversion_rate"]
    )

    integer_columns = count_columns
    result[integer_columns] = result[integer_columns].astype(int)
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _normalise_channels(
    channels: Mapping[str, Mapping[str, float]] | None,
) -> pd.DataFrame:
    source = channels or DEFAULT_CHANNELS
    rows = []
    for channel, values in source.items():
        try:
            cost = float(values["cost_per_contact"])
            multiplier = float(values["conversion_multiplier"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid channel configuration for {channel!r}") from exc
        rows.append(
            {
                "channel": str(channel),
                "cost_per_contact": cost,
                "conversion_multiplier": multiplier,
            }
        )
    if not rows:
        raise ValueError("channels must contain at least one channel")
    return pd.DataFrame(rows)


def _audience_cells(profile: pd.DataFrame, max_customers: int) -> pd.DataFrame:
    _require_columns(
        profile,
        {"ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu"},
        "profile",
    )
    working = profile.copy()
    working["predicted_arpu"] = pd.to_numeric(working["predicted_arpu"], errors="coerce")
    working = working.dropna(subset=["current_tariff", "arpu_segment", "predicted_arpu"])
    working["current_tariff"] = working["current_tariff"].astype(str)
    working = working[working["arpu_segment"].isin(ARPU_LABELS)]
    working = working.sort_values("ID_NUMBER", kind="stable")

    rows = []
    for (current_tariff, arpu_segment), segment in working.groupby(
        ["current_tariff", "arpu_segment"], observed=True, sort=False
    ):
        contacted = segment.head(max_customers)
        rows.append(
            {
                "tariff_plan_code_from": current_tariff,
                "arpu_segment": arpu_segment,
                "audience_available": int(len(segment)),
                "n_customers": int(len(contacted)),
                "predicted_arpu_mean": float(contacted["predicted_arpu"].mean()),
                "predicted_arpu_sum": float(contacted["predicted_arpu"].sum()),
                "capped_at_campaign_limit": bool(len(segment) > max_customers),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_table(
    profile: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    channels: Mapping[str, Mapping[str, float]] | None = None,
    config: PriorConfig = PriorConfig(),
) -> pd.DataFrame:
    """Expand transition priors into channel-specific campaign candidates."""

    _require_columns(
        priors,
        {
            "tariff_plan_code_from",
            "arpu_segment",
            "tariff_plan_code_to",
            "smoothed_change_pct",
            "smoothed_conversion_rate",
            "prior_standard_error",
            "history_confidence",
        },
        "priors",
    )
    audiences = _audience_cells(profile, config.max_customers_per_campaign)
    result = audiences.merge(
        priors,
        on=["tariff_plan_code_from", "arpu_segment"],
        how="inner",
        validate="one_to_many",
    )
    channel_table = _normalise_channels(channels)
    result = result.merge(channel_table, how="cross")

    result["effective_conversion_rate"] = (
        result["smoothed_conversion_rate"] * result["conversion_multiplier"]
    ).clip(upper=1.0)
    result["expected_lift_ratio"] = (
        result["smoothed_change_pct"] * result["effective_conversion_rate"]
    )
    result["conservative_change_pct"] = (
        result["smoothed_change_pct"]
        - config.conservative_z * result["prior_standard_error"]
    ).clip(config.change_clip_low, config.change_clip_high)
    result["optimistic_change_pct"] = (
        result["smoothed_change_pct"]
        + config.conservative_z * result["prior_standard_error"]
    ).clip(config.change_clip_low, config.change_clip_high)
    result["conservative_lift_ratio"] = (
        result["conservative_change_pct"] * result["effective_conversion_rate"]
    )
    result["optimistic_lift_ratio"] = (
        result["optimistic_change_pct"] * result["effective_conversion_rate"]
    )

    result["contact_cost"] = result["n_customers"] * result["cost_per_contact"]
    result["expected_gross_lift"] = (
        result["predicted_arpu_sum"] * result["expected_lift_ratio"]
    )
    result["expected_net_gain"] = result["expected_gross_lift"] - result["contact_cost"]
    result["conservative_net_gain"] = (
        result["predicted_arpu_sum"] * result["conservative_lift_ratio"]
        - result["contact_cost"]
    )
    result["optimistic_net_gain"] = (
        result["predicted_arpu_sum"] * result["optimistic_lift_ratio"]
        - result["contact_cost"]
    )
    denominator = result["n_customers"].replace(0, np.nan)
    result["expected_net_per_contact"] = result["expected_net_gain"] / denominator
    result["conservative_net_per_contact"] = result["conservative_net_gain"] / denominator
    result["uncertainty_value"] = (
        result["optimistic_net_gain"] - result["conservative_net_gain"]
    ) / 2.0

    result["campaign_name"] = (
        "prior_"
        + result["tariff_plan_code_from"].astype(str)
        + "_"
        + result["arpu_segment"].astype(str).str.lower()
        + "_to_"
        + result["tariff_plan_code_to"].astype(str)
        + "_"
        + result["channel"].astype(str)
    )
    result["filter_current_tariff"] = result["tariff_plan_code_from"]
    result["filter_arpu_segment"] = result["arpu_segment"]
    result["target_tariff"] = result["tariff_plan_code_to"]

    return result.sort_values(
        ["conservative_net_per_contact", "expected_net_gain", "history_confidence"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def select_pilot_candidates(
    candidates: pd.DataFrame,
    *,
    top_k: int = 20,
    pilot_channel: str = "sms",
    min_customers: int = 10,
    max_per_audience: int = 2,
) -> pd.DataFrame:
    """Select a diverse shortlist of transitions for pilot exploration.

    The shortlist balances expected upside with uncertainty.  A fixed pilot
    channel makes observations easier to compare and avoids spending pilot
    budget on calls.  The agent remains free to choose a different final
    channel after observing the pilots.
    """

    _require_columns(
        candidates,
        {
            "channel",
            "n_customers",
            "tariff_plan_code_from",
            "arpu_segment",
            "tariff_plan_code_to",
            "expected_net_gain",
            "uncertainty_value",
        },
        "candidates",
    )
    if top_k <= 0:
        return candidates.head(0).copy()

    pool = candidates[
        (candidates["channel"] == pilot_channel)
        & (candidates["n_customers"] >= min_customers)
    ].copy()
    if pool.empty:
        return pool

    # Optimism is useful during exploration, but only half of the uncertainty
    # band is added so weak fallbacks cannot dominate solid historical signals.
    pool["pilot_priority_score"] = (
        pool["expected_net_gain"] + 0.5 * pool["uncertainty_value"]
    )
    pool = pool.sort_values(
        ["pilot_priority_score", "history_confidence"],
        ascending=[False, False],
        kind="stable",
    )

    selected_indices: list[int] = []
    audience_counts: dict[tuple[str, str], int] = {}
    for idx, row in pool.iterrows():
        audience = (str(row["tariff_plan_code_from"]), str(row["arpu_segment"]))
        if audience_counts.get(audience, 0) >= max_per_audience:
            continue
        selected_indices.append(idx)
        audience_counts[audience] = audience_counts.get(audience, 0) + 1
        if len(selected_indices) >= top_k:
            break

    return pool.loc[selected_indices].reset_index(drop=True)


def build_default_candidates(
    profile: pd.DataFrame,
    change_tariff: pd.DataFrame,
    tariffs: pd.DataFrame,
    *,
    channels: Mapping[str, Mapping[str, float]] | None = None,
    config: PriorConfig = PriorConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning ``(priors, candidates)``."""

    priors = build_transition_priors(change_tariff, tariffs, config=config)
    candidates = build_candidate_table(
        profile,
        priors,
        channels=channels,
        config=config,
    )
    return priors, candidates


def build_candidates(
    profile: pd.DataFrame,
    tariffs: pd.DataFrame,
    base_dir: str | Path,
    *,
    config: PriorConfig = PriorConfig(),
) -> pd.DataFrame:
    """Adapter for the stable contract consumed by :mod:`strategy`.

    ``strategy.CampaignStrategy`` deliberately knows nothing about historical
    file layouts.  It calls this function with the live profile, tariff table
    and repository root.  The returned ``prior_mean`` is channel-neutral, so a
    pilot observation can be divided by its channel multiplier and combined
    with the prior on the same scale.
    """

    history_path = Path(base_dir) / "data" / "change_tariff.csv"
    history = pd.read_csv(history_path)
    priors = build_transition_priors(history, tariffs, config=config)

    _require_columns(
        profile,
        {"ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu"},
        "profile",
    )
    cells = profile.dropna(
        subset=["current_tariff", "arpu_segment", "predicted_arpu"]
    ).copy()
    cells["predicted_arpu"] = pd.to_numeric(cells["predicted_arpu"], errors="coerce")
    cells = cells.dropna(subset=["predicted_arpu"])
    cells = (
        cells.groupby(["current_tariff", "arpu_segment"], observed=True)
        .agg(segment_arpu_sum=("predicted_arpu", "sum"))
        .reset_index()
        .rename(columns={"current_tariff": "tariff_plan_code_from"})
    )

    result = priors.merge(
        cells,
        on=["tariff_plan_code_from", "arpu_segment"],
        how="inner",
        validate="many_to_one",
    )
    result["prior_mean"] = result["expected_base_lift_ratio"].clip(-1.0, 3.0)

    # Lift uncertainty is on the ARPU-change scale.  Convert it to the same
    # channel-neutral expected-lift scale as prior_mean.  CampaignStrategy also
    # applies a 0.20 floor because the judging population differs from history.
    result["prior_std"] = (
        result["prior_standard_error"] * result["smoothed_conversion_rate"]
    ).clip(0.05, 2.0)

    # Observed transitions lead the queue.  Missing pairs remain available for
    # exploration, but a large price gap alone must not crowd out real evidence.
    source_weight = result["prior_source"].map(
        {
            "observed_smoothed": 1.00,
            "target_history_fallback": 0.35,
            "tariff_price_fallback": 0.15,
        }
    ).fillna(0.10)
    evidence_weight = 0.25 + 0.75 * result["history_confidence"]
    value_per_arpu = result["prior_mean"].clip(lower=0.0) + 0.02
    result["priority"] = (
        result["segment_arpu_sum"]
        * value_per_arpu
        * source_weight
        * evidence_weight
    )
    result["source"] = result["prior_source"]

    return (
        result.rename(
            columns={
                "tariff_plan_code_from": "current_tariff",
                "tariff_plan_code_to": "target_tariff",
            }
        )[
            [
                "current_tariff",
                "arpu_segment",
                "target_tariff",
                "prior_mean",
                "prior_std",
                "priority",
                "source",
                "n_observations",
                "history_confidence",
            ]
        ]
        .sort_values(["priority", "history_confidence"], ascending=False, kind="stable")
        .reset_index(drop=True)
    )


__all__ = [
    "ARPU_BINS",
    "ARPU_LABELS",
    "DEFAULT_CHANNELS",
    "PriorConfig",
    "build_candidate_table",
    "build_candidates",
    "build_default_candidates",
    "build_transition_priors",
    "fallback_transition_prior",
    "select_pilot_candidates",
]
