import tempfile
import unittest
from pathlib import Path

import pandas as pd

from visual_report import generate_visual_report


class VisualReportTest(unittest.TestCase):
    def test_report_contains_consistent_contact_details(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = generate_visual_report(seed=42, output_dir=directory)

            report_path = Path(artifacts["report_path"])
            ledger = pd.read_csv(artifacts["ledger_path"])
            subscribers = pd.read_csv(artifacts["subscribers_path"])
            campaigns = pd.read_csv(artifacts["campaigns_path"])

            self.assertTrue(report_path.is_file())
            self.assertIn("Что означают 23 441 абонент", report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(ledger), artifacts["result"]["total_contacts"])
            self.assertEqual(len(subscribers), 23_441)
            self.assertEqual(
                len(campaigns),
                artifacts["n_pilots"] + artifacts["n_final_campaigns"],
            )
            self.assertLessEqual(artifacts["n_final_campaigns"], 10)


if __name__ == "__main__":
    unittest.main()
