import unittest

import numpy as np
import pandas as pd

from priors import (
    DEFAULT_CHANNELS,
    build_candidate_table,
    build_candidates,
    build_transition_priors,
    fallback_transition_prior,
    select_pilot_candidates,
)


class PriorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = pd.read_csv("customer_profile.csv")
        cls.history = pd.read_csv("data/change_tariff.csv")
        cls.tariffs = pd.read_csv("data/dict_tariff.csv")
        cls.priors = build_transition_priors(cls.history, cls.tariffs)
        cls.candidates = build_candidate_table(cls.profile, cls.priors)

    def test_prior_grid_is_complete_and_finite(self):
        n_tariffs = self.tariffs["tariff_plan_code"].nunique()
        self.assertEqual(len(self.priors), n_tariffs * 3 * (n_tariffs - 1))
        self.assertFalse(
            (
                self.priors["tariff_plan_code_from"]
                == self.priors["tariff_plan_code_to"]
            ).any()
        )
        numeric = self.priors[
            [
                "smoothed_change_pct",
                "smoothed_conversion_rate",
                "expected_base_lift_ratio",
                "prior_standard_error",
            ]
        ].to_numpy()
        self.assertTrue(np.isfinite(numeric).all())

    def test_history_and_fallback_rows_are_both_present(self):
        sources = set(self.priors["prior_source"])
        self.assertIn("observed_smoothed", sources)
        self.assertTrue(
            bool(
                sources
                & {"target_history_fallback", "tariff_price_fallback"}
            )
        )
        self.assertTrue((self.priors["history_confidence"] >= 0).all())
        self.assertTrue((self.priors["history_confidence"] < 1).all())

    def test_candidate_table_matches_live_audiences_and_channels(self):
        audience_cells = (
            self.profile.dropna(subset=["current_tariff", "arpu_segment"])[
                ["current_tariff", "arpu_segment"]
            ]
            .drop_duplicates()
            .shape[0]
        )
        n_tariffs = self.tariffs["tariff_plan_code"].nunique()
        expected_rows = audience_cells * (n_tariffs - 1) * len(DEFAULT_CHANNELS)
        self.assertEqual(len(self.candidates), expected_rows)
        self.assertLessEqual(int(self.candidates["n_customers"].max()), 5_000)
        self.assertFalse(
            (
                self.candidates["filter_current_tariff"]
                == self.candidates["target_tariff"]
            ).any()
        )

    def test_fallback_uses_tariff_price_direction(self):
        cheap_to_expensive = fallback_transition_prior(
            "tariff_1", "tariff_12", self.tariffs
        )
        expensive_to_cheap = fallback_transition_prior(
            "tariff_12", "tariff_1", self.tariffs
        )
        self.assertGreater(cheap_to_expensive["smoothed_change_pct"], 0)
        self.assertLess(expensive_to_cheap["smoothed_change_pct"], 0)
        self.assertEqual(cheap_to_expensive["history_confidence"], 0.0)

    def test_pilot_shortlist_is_diverse(self):
        shortlist = select_pilot_candidates(
            self.candidates,
            top_k=20,
            pilot_channel="sms",
            max_per_audience=2,
        )
        self.assertEqual(len(shortlist), 20)
        self.assertEqual(set(shortlist["channel"]), {"sms"})
        per_audience = shortlist.groupby(
            ["tariff_plan_code_from", "arpu_segment"]
        ).size()
        self.assertLessEqual(int(per_audience.max()), 2)

    def test_strategy_contract_is_available(self):
        frame = build_candidates(self.profile, self.tariffs, ".")
        required = {
            "current_tariff",
            "arpu_segment",
            "target_tariff",
            "prior_mean",
            "prior_std",
            "priority",
            "source",
        }
        self.assertTrue(required.issubset(frame.columns))
        self.assertFalse(frame[list(required)].isna().any().any())
        self.assertFalse((frame["current_tariff"] == frame["target_tariff"]).any())
        self.assertEqual(len(frame), len(self.priors))


if __name__ == "__main__":
    unittest.main()
