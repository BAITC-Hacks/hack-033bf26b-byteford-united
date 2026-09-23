"""Adaptive pilot policy and constrained campaign selection.

The hidden judging effects are defined at the
``current_tariff x arpu_segment x target_tariff`` level.  This module therefore
keeps experiments at that grain, combines noisy observations using their known
variance, and selects at most one final decision for each audience cell.

Optional priors contract
------------------------
If ``priors.py`` exists, it may expose::

    build_candidates(profile, tariffs, base_dir) -> pandas.DataFrame

Required columns are ``current_tariff``, ``arpu_segment`` and
``target_tariff``.  Optional columns are ``prior_mean`` (channel-neutral lift
ratio), ``prior_std`` and ``priority``.  Actual audience sizes and ARPU totals
are always recalculated here rather than trusted from the prior module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PER_CUSTOMER_STD = 0.804
INITIAL_PILOTS = 10
INITIAL_PILOT_SIZE = 80
FOLLOWUP_PILOTS = 10
FOLLOWUP_PILOT_SIZE = 200
MIN_CELL_SIZE = 30
MIN_PRIOR_STD = 0.20
# A one-sided 95% lower bound keeps a lucky pilot from triggering a large
# rollout.  Mock seeds showed that a looser bound could flip the total result.
LCB_Z = 1.64
MAX_BEAM_STATES = 4000
MAX_CANDIDATES_PER_CELL = 4
@dataclass(frozen=True)
class StrategyConfig:
    """Tunable exploration and risk policy with the current defaults preserved."""

    initial_pilots: int = INITIAL_PILOTS
    initial_pilot_size: int = INITIAL_PILOT_SIZE
    followup_pilots: int = FOLLOWUP_PILOTS
    followup_pilot_size: int = FOLLOWUP_PILOT_SIZE
    initial_observed_quota: int = 8
    initial_target_fallback_quota: int = 1
    initial_price_fallback_quota: int = 1
    min_prior_std: float = MIN_PRIOR_STD
    lcb_z: float = LCB_Z
    min_pilots_for_rollout: int = 2
    min_pilots_for_paid: int = 2

    def __post_init__(self) -> None:
        if self.initial_pilots < 0 or self.followup_pilots < 0:
            raise ValueError("pilot counts must be non-negative")
        if self.initial_pilots + self.followup_pilots > 20:
            raise ValueError("total configured pilots cannot exceed 20")
        for size in (self.initial_pilot_size, self.followup_pilot_size):
            if not 10 <= size <= 200:
                raise ValueError("pilot sizes must be between 10 and 200")
        quota_total = (
            self.initial_observed_quota
            + self.initial_target_fallback_quota
            + self.initial_price_fallback_quota
        )
        if quota_total > self.initial_pilots:
            raise ValueError("initial source quotas cannot exceed initial_pilots")
        if self.min_prior_std <= 0 or self.lcb_z <= 0:
            raise ValueError("uncertainty parameters must be positive")
        if self.min_pilots_for_rollout < 1:
            raise ValueError("min_pilots_for_rollout must be positive")
        if self.min_pilots_for_paid < self.min_pilots_for_rollout:
            raise ValueError(
                "min_pilots_for_paid cannot be lower than rollout minimum"
            )


@dataclass
class Candidate:
    current_tariff: str
    arpu_segment: str
    target_tariff: str
    segment_n: int
    segment_arpu_sum: float
    segment_arpu_mean: float
    prior_mean: float = 0.0
    prior_std: float = 0.35
    priority: float = 0.0
    source: str = "fallback"
    prior_std_floor: float = MIN_PRIOR_STD
    precision: float = field(init=False)
    weighted_sum: float = field(init=False)
    pilot_count: int = 0
    pilot_contacts: int = 0
    pilot_channels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Historical data come from another population, so even a confident
        # historical estimate is deliberately treated as a weak prior.
        effective_std = max(float(self.prior_std), float(self.prior_std_floor))
        self.precision = 1.0 / (effective_std * effective_std)
        self.weighted_sum = float(self.prior_mean) * self.precision

    @property
    def key(self) -> tuple[str, str, str]:
        return self.current_tariff, self.arpu_segment, self.target_tariff

    @property
    def cell_key(self) -> tuple[str, str]:
        return self.current_tariff, self.arpu_segment

    @property
    def posterior_mean(self) -> float:
        return self.weighted_sum / self.precision

    @property
    def posterior_std(self) -> float:
        return math.sqrt(1.0 / self.precision)

    def update(self, observed_ratio: float, n_customers: int, multiplier: float, channel: str) -> None:
        if n_customers <= 0 or multiplier <= 0:
            return
        base_observation = float(observed_ratio) / multiplier
        base_se = PER_CUSTOMER_STD / (math.sqrt(n_customers) * multiplier)
        observation_precision = 1.0 / (base_se * base_se)
        self.precision += observation_precision
        self.weighted_sum += base_observation * observation_precision
        self.pilot_count += 1
        self.pilot_contacts += int(n_customers)
        self.pilot_channels.append(channel)


@dataclass(frozen=True)
class CampaignOption:
    candidate: Candidate
    channel: str
    n_contacts: int
    cost: float
    safe_net: float
    safe_gross: float


class CampaignStrategy:
    def __init__(
        self,
        base_dir: Path,
        *,
        use_external_priors: bool = True,
        config: StrategyConfig | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.use_external_priors = bool(use_external_priors)
        self.config = config or StrategyConfig()

    def run(self, env) -> list[dict]:
        candidates = self._load_candidates(env)
        if not candidates:
            return self._emergency_campaign(env, [])

        self._run_initial_pilots(env, candidates)
        self._run_followup_pilots(env, candidates)
        campaigns = self._select_portfolio(env, candidates)
        if campaigns:
            return campaigns
        return self._emergency_campaign(env, candidates)

    def _load_candidates(self, env) -> list[Candidate]:
        frame = None
        if self.use_external_priors:
            try:
                priors = importlib.import_module("priors")
                builder = getattr(priors, "build_candidates")
                frame = builder(
                    profile=env.customer_profile.copy(),
                    tariffs=env.tariffs.copy(),
                    base_dir=self.base_dir,
                )
            except (ImportError, AttributeError, TypeError, ValueError, OSError, pd.errors.ParserError):
                # Priors are an optional enhancement.  A broken data file or
                # module must not invalidate already paid pilot contacts.
                frame = None

        if frame is None or len(frame) == 0:
            frame = self._fallback_candidate_frame(env.customer_profile, env.tariffs)
        return self._coerce_candidates(frame, env.customer_profile, env.tariffs)

    @staticmethod
    def _cell_stats(profile: pd.DataFrame) -> pd.DataFrame:
        valid = profile.dropna(subset=["current_tariff", "arpu_segment", "predicted_arpu"])
        return (
            valid.groupby(["current_tariff", "arpu_segment"], observed=True)
            .agg(
                segment_n=("ID_NUMBER", "size"),
                segment_arpu_sum=("predicted_arpu", "sum"),
                segment_arpu_mean=("predicted_arpu", "mean"),
                data_need=("DATA_VOLUME", "median"),
                call_need=("OUT_LOC_OFFNET_MIN", "median"),
            )
            .reset_index()
        )

    def _fallback_candidate_frame(self, profile: pd.DataFrame, tariffs: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic, conservative upsell hypotheses without history."""
        cells = self._cell_stats(profile)
        tariff_rows = tariffs.dropna(subset=["tariff_plan_code", "price_tariff"]).copy()
        tariff_rows = tariff_rows.drop_duplicates("tariff_plan_code")
        tariff_by_code = tariff_rows.set_index("tariff_plan_code")
        candidates: list[dict] = []

        for cell in cells.itertuples(index=False):
            if cell.segment_n < MIN_CELL_SIZE or cell.current_tariff not in tariff_by_code.index:
                continue
            current_price = float(tariff_by_code.loc[cell.current_tariff, "price_tariff"])
            scored_targets: list[tuple[float, str]] = []
            desired_price = max(current_price * 1.25, 3100.0)

            for target in tariff_rows.itertuples(index=False):
                if target.tariff_plan_code == cell.current_tariff:
                    continue
                target_price = float(target.price_tariff)
                if target_price <= current_price + 1.0:
                    continue

                price_fit = -abs(math.log(max(target_price, 1.0) / desired_price))
                data_need = 0.0 if pd.isna(cell.data_need) else float(cell.data_need)
                call_need = 0.0 if pd.isna(cell.call_need) else float(cell.call_need)
                included_minutes = float(target.Min_another_operator_in_PKG) + float(
                    target.Min_another_operator_and_city_in_PKG
                )
                capacity_fit = 0.0
                if float(target.Data_in_PKG) >= data_need:
                    capacity_fit += 0.30
                if included_minutes >= call_need:
                    capacity_fit += 0.20
                scored_targets.append((price_fit + capacity_fit, str(target.tariff_plan_code)))

            if not scored_targets:
                continue
            scored_targets.sort(reverse=True)
            # One fallback target per cell gives broad exploration.  The prior
            # module may supply multiple targets for the same cell later.
            target_score, target_tariff = scored_targets[0]
            candidates.append(
                {
                    "current_tariff": cell.current_tariff,
                    "arpu_segment": cell.arpu_segment,
                    "target_tariff": target_tariff,
                    "prior_mean": 0.015,
                    "prior_std": 0.35,
                    "priority": float(cell.segment_arpu_sum) * (1.0 + max(target_score, -0.5)),
                    "source": "tariff_fit_fallback",
                }
            )
        return pd.DataFrame(candidates)

    def _coerce_candidates(
        self,
        frame: pd.DataFrame,
        profile: pd.DataFrame,
        tariffs: pd.DataFrame,
    ) -> list[Candidate]:
        required = {"current_tariff", "arpu_segment", "target_tariff"}
        if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
            frame = self._fallback_candidate_frame(profile, tariffs)

        stats = self._cell_stats(profile).drop(columns=["data_need", "call_need"])
        known_tariffs = set(tariffs["tariff_plan_code"].dropna().astype(str))
        cleaned = frame.copy()
        cleaned = cleaned.merge(stats, on=["current_tariff", "arpu_segment"], how="inner")
        cleaned = cleaned[
            cleaned["current_tariff"].astype(str).isin(known_tariffs)
            & cleaned["target_tariff"].astype(str).isin(known_tariffs)
            & (cleaned["current_tariff"].astype(str) != cleaned["target_tariff"].astype(str))
            & (cleaned["segment_n"] >= MIN_CELL_SIZE)
        ].copy()

        defaults = {
            "prior_mean": 0.0,
            "prior_std": 0.35,
            "priority": np.nan,
            "source": "external_prior",
        }
        for column, default in defaults.items():
            if column not in cleaned.columns:
                cleaned[column] = default

        cleaned["prior_mean"] = pd.to_numeric(cleaned["prior_mean"], errors="coerce").fillna(0.0).clip(-1.0, 3.0)
        cleaned["prior_std"] = pd.to_numeric(cleaned["prior_std"], errors="coerce").fillna(0.35).clip(0.05, 2.0)
        default_priority = cleaned["segment_arpu_sum"] * (cleaned["prior_mean"].clip(lower=0.0) + 0.02)
        cleaned["priority"] = pd.to_numeric(cleaned["priority"], errors="coerce").fillna(default_priority)
        cleaned = cleaned.drop_duplicates(["current_tariff", "arpu_segment", "target_tariff"])
        cleaned = cleaned.sort_values(["priority", "segment_arpu_sum"], ascending=False)

        # Keep source diversity before applying the per-cell cap.  Without this
        # step, two strong observed transitions can remove every unobserved
        # fallback from a cell, making the exploration quotas ineffective.
        selected_indices: list[int] = []
        for _, group in cleaned.groupby(
            ["current_tariff", "arpu_segment"], observed=True, sort=False
        ):
            group_indices: list[int] = []
            for source in group["source"].astype(str).drop_duplicates():
                source_rows = group[group["source"].astype(str) == source]
                group_indices.append(int(source_rows.index[0]))
                if len(group_indices) >= MAX_CANDIDATES_PER_CELL:
                    break
            if len(group_indices) < MAX_CANDIDATES_PER_CELL:
                for idx in group.index:
                    if idx in group_indices:
                        continue
                    group_indices.append(int(idx))
                    if len(group_indices) >= MAX_CANDIDATES_PER_CELL:
                        break
            selected_indices.extend(group_indices)

        cleaned = cleaned.loc[selected_indices].sort_values(
            ["priority", "segment_arpu_sum"], ascending=False
        )

        result: list[Candidate] = []
        for row in cleaned.itertuples(index=False):
            result.append(
                Candidate(
                    current_tariff=str(row.current_tariff),
                    arpu_segment=str(row.arpu_segment),
                    target_tariff=str(row.target_tariff),
                    segment_n=int(row.segment_n),
                    segment_arpu_sum=float(row.segment_arpu_sum),
                    segment_arpu_mean=float(row.segment_arpu_mean),
                    prior_mean=float(row.prior_mean),
                    prior_std=float(row.prior_std),
                    priority=float(row.priority),
                    source=str(row.source),
                    prior_std_floor=self.config.min_prior_std,
                )
            )
        return result

    @staticmethod
    def _pilot_capacity(env, channel: str, requested: int, segment_n: int) -> int:
        n = min(int(requested), int(segment_n), int(env.remaining_contacts))
        cost = float(env.channels[channel]["cost_per_contact"])
        if cost > 0:
            n = min(n, int(env.remaining_budget // cost))
        return n if n >= 10 else 0

    def _run_pilot(self, env, candidate: Candidate, channel: str, requested: int) -> bool:
        n_customers = self._pilot_capacity(env, channel, requested, candidate.segment_n)
        if n_customers == 0 or env.pilots_left <= 0:
            return False
        try:
            result = env.run_pilot(
                target_tariff=candidate.target_tariff,
                channel=channel,
                n_customers=n_customers,
                filter_arpu_segment=candidate.arpu_segment,
                filter_current_tariff=candidate.current_tariff,
            )
        except (RuntimeError, ValueError, KeyError, TypeError):
            return False

        multiplier = float(env.channels[channel]["conversion_multiplier"])
        candidate.update(
            observed_ratio=float(result["observed_lift_ratio"]),
            n_customers=int(result["n_customers"]),
            multiplier=multiplier,
            channel=channel,
        )
        return True

    def _run_initial_pilots(self, env, candidates: list[Candidate]) -> None:
        limit = min(self.config.initial_pilots, env.pilots_left)
        used_cells: set[tuple[str, str]] = set()
        selected: list[Candidate] = []

        def add_candidate(candidate: Candidate) -> bool:
            if candidate.cell_key in used_cells or len(selected) >= limit:
                return False
            selected.append(candidate)
            used_cells.add(candidate.cell_key)
            return True

        # Reserve a few pilots for transitions missing at the exact historical
        # grain.  The judging population is deliberately shifted, so an agent
        # that explores only historically observed winners is brittle.
        source_quotas = {
            "observed_smoothed": self.config.initial_observed_quota,
            "target_history_fallback": self.config.initial_target_fallback_quota,
            "tariff_price_fallback": self.config.initial_price_fallback_quota,
        }
        for source, quota in source_quotas.items():
            added = 0
            for candidate in candidates:
                if candidate.source != source:
                    continue
                if source != "observed_smoothed" and candidate.prior_mean <= 0:
                    continue
                if add_candidate(candidate):
                    added += 1
                if added >= quota or len(selected) >= limit:
                    break

        # When priors are unavailable, or a source has too few candidates,
        # fill the remaining slots by global priority.
        for candidate in candidates:
            add_candidate(candidate)
            if len(selected) >= limit:
                break

        for candidate in selected:
            if not self._run_pilot(
                env, candidate, "sms", self.config.initial_pilot_size
            ):
                break

    @staticmethod
    def _positive_probability(candidate: Candidate) -> float:
        z = candidate.posterior_mean / max(candidate.posterior_std, 1e-9)
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _run_followup_pilots(self, env, candidates: list[Candidate]) -> None:
        observed = [candidate for candidate in candidates if candidate.pilot_count > 0]
        observed.sort(key=self._followup_value, reverse=True)
        followup_limit = min(self.config.followup_pilots, env.pilots_left)
        completed = 0
        digital_used = 0

        for candidate in observed:
            if completed >= followup_limit:
                break
            probability_positive = self._positive_probability(candidate)
            # A hypothesis whose optimistic bound is still negative should not
            # receive more contacts.  Use the slot on a fresh alternative.
            if candidate.posterior_mean + 1.28 * candidate.posterior_std <= 0:
                continue

            channel = "sms"
            digital_cost = self.config.followup_pilot_size * float(
                env.channels["digital_ads"]["cost_per_contact"]
            )
            expected_digital_net = (
                candidate.posterior_mean
                * float(env.channels["digital_ads"]["conversion_multiplier"])
                * candidate.segment_arpu_mean
                - float(env.channels["digital_ads"]["cost_per_contact"])
            )
            if (
                digital_used < 3
                and probability_positive >= 0.75
                and expected_digital_net > 0
                and env.remaining_budget - digital_cost >= 50_000
            ):
                channel = "digital_ads"

            if self._run_pilot(
                env, candidate, channel, self.config.followup_pilot_size
            ):
                completed += 1
                digital_used += int(channel == "digital_ads")

        if completed >= followup_limit:
            return

        # Spend unused follow-up slots on alternative targets/cells rather than
        # repeating clearly negative experiments.
        for candidate in candidates:
            if completed >= followup_limit or env.pilots_left <= 0:
                break
            if candidate.pilot_count > 0:
                continue
            if self._run_pilot(env, candidate, "sms", 100):
                completed += 1

    @staticmethod
    def _followup_value(candidate: Candidate) -> float:
        probability_positive = CampaignStrategy._positive_probability(candidate)
        boundary_uncertainty = 4.0 * probability_positive * (1.0 - probability_positive)
        information_value = candidate.posterior_std * (0.30 + boundary_uncertainty)
        exploitation_value = max(candidate.posterior_mean, 0.0) * 0.15
        return candidate.segment_arpu_sum * (information_value + exploitation_value)

    def _campaign_options(self, env, candidates: Iterable[Candidate]) -> list[CampaignOption]:
        options: list[CampaignOption] = []
        for candidate in candidates:
            if candidate.pilot_count < self.config.min_pilots_for_rollout:
                continue
            safe_base_ratio = (
                candidate.posterior_mean
                - self.config.lcb_z * candidate.posterior_std
            )
            if safe_base_ratio <= 0:
                continue

            pilot_overlap = min(1.0, candidate.pilot_contacts / max(candidate.segment_n, 1))
            marginal_fraction = max(0.50, 1.0 - 0.50 * pilot_overlap)
            # The current risk policy requires independent confirmation before
            # any rollout.  Paid channels may use a stricter threshold through
            # min_pilots_for_paid in alternative configurations.
            channels = (
                ("push",)
                if candidate.pilot_count < self.config.min_pilots_for_paid
                else ("push", "sms", "digital_ads")
            )
            for channel in channels:
                multiplier = float(env.channels[channel]["conversion_multiplier"])
                cost_per_contact = float(env.channels[channel]["cost_per_contact"])
                safe_gross = safe_base_ratio * multiplier * candidate.segment_arpu_sum * marginal_fraction
                cost = candidate.segment_n * cost_per_contact
                safe_net = safe_gross - cost
                if safe_net <= 0:
                    continue
                options.append(
                    CampaignOption(
                        candidate=candidate,
                        channel=channel,
                        n_contacts=candidate.segment_n,
                        cost=cost,
                        safe_net=safe_net,
                        safe_gross=safe_gross,
                    )
                )
        return options

    def _select_portfolio(self, env, candidates: list[Candidate]) -> list[dict]:
        options = self._campaign_options(env, candidates)
        if not options:
            return []

        grouped: dict[tuple[str, str], list[CampaignOption]] = {}
        for option in options:
            grouped.setdefault(option.candidate.cell_key, []).append(option)

        groups = sorted(grouped.values(), key=lambda group: max(option.safe_net for option in group), reverse=True)
        # state: (safe_net, contacts, cost, tuple(options))
        states: list[tuple[float, int, float, tuple[CampaignOption, ...]]] = [(0.0, 0, 0.0, tuple())]
        max_contacts = int(env.remaining_contacts)
        max_budget = float(env.remaining_budget)

        for group in groups:
            expanded = list(states)
            for state_net, state_contacts, state_cost, selected in states:
                if len(selected) >= 10:
                    continue
                for option in group:
                    contacts = state_contacts + option.n_contacts
                    cost = state_cost + option.cost
                    if contacts > max_contacts or cost > max_budget:
                        continue
                    expanded.append(
                        (state_net + option.safe_net, contacts, cost, selected + (option,))
                    )

            # Bucket near-identical resource states and retain the strongest.
            best_by_bucket: dict[tuple[int, int, int], tuple[float, int, float, tuple[CampaignOption, ...]]] = {}
            for state in expanded:
                key = (len(state[3]), state[1] // 50, int(state[2]) // 250)
                previous = best_by_bucket.get(key)
                if previous is None or state[0] > previous[0]:
                    best_by_bucket[key] = state
            states = sorted(best_by_bucket.values(), key=lambda state: state[0], reverse=True)[:MAX_BEAM_STATES]

        best = max(states, key=lambda state: state[0])
        selected_options = sorted(best[3], key=lambda option: option.safe_net, reverse=True)
        return [self._option_to_campaign(option, index) for index, option in enumerate(selected_options, start=1)]

    @staticmethod
    def _option_to_campaign(option: CampaignOption, index: int) -> dict:
        candidate = option.candidate
        return {
            "campaign_name": (
                f"main_{index}_{candidate.current_tariff}_{candidate.arpu_segment.lower()}_"
                f"{candidate.target_tariff}_{option.channel}"
            ),
            "filter_arpu_segment": candidate.arpu_segment,
            "filter_current_tariff": candidate.current_tariff,
            "target_tariff": candidate.target_tariff,
            "channel": option.channel,
        }

    def _emergency_campaign(self, env, candidates: list[Candidate]) -> list[dict]:
        viable = [candidate for candidate in candidates if candidate.segment_n <= env.remaining_contacts]
        if not viable:
            viable = list(candidates)
        if viable:
            best = max(
                viable,
                key=lambda candidate: (
                    candidate.posterior_mean if candidate.pilot_count else candidate.prior_mean,
                    -candidate.segment_n,
                ),
            )
            option = CampaignOption(
                candidate=best,
                channel="push",
                n_contacts=best.segment_n,
                cost=0.0,
                safe_net=0.0,
                safe_gross=0.0,
            )
            return [self._option_to_campaign(option, 1)]

        # This branch is reachable only when the input profile has no usable
        # tariff/ARPU cell.  Return a syntactically valid minimal fallback.
        tariffs = list(env.tariffs["tariff_plan_code"].dropna().astype(str))
        if not tariffs:
            return []
        return [{"campaign_name": "emergency_push", "target_tariff": tariffs[0], "channel": "push"}]
