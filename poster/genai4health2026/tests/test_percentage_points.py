"""Checks of AUC-scale versus percentage-point display, including the real bundle.

Read only: python tests/test_percentage_points.py from the poster folder.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_preview as poster


class PercentagePoints(unittest.TestCase):
    def test_sign_scale_and_precision(self):
        self.assertEqual(poster.percentage_points(.012), "+1.20")
        self.assertEqual(poster.percentage_points(-.0168), "-1.68")
        self.assertEqual(poster.percentage_points(.007046286892287812), "+0.70")
        self.assertEqual(poster.percentage_points(.002590397555366275), "+0.26")
        self.assertEqual(poster.percentage_points(.01234, signed=False), "1.23")

    def test_reject_nonfinite(self):
        for value in (float("nan"), float("inf"), None, True):
            with self.assertRaises(ValueError):
                poster.percentage_points(value)

    def test_intervals_and_missing_endpoint(self):
        values = poster.poster_pp_values({"numeric": {}, "full_precision": {}})
        self.assertEqual(values["DEnvelopeRandomEpFiftyCI"], r"[+0.68,\,+1.73]")
        self.assertEqual(values["DOracleRandomEpFiftyCI"], r"[+0.51,\,+1.49]")
        self.assertEqual(values["DCoverRandomEpHundredCI"], r"[-2.33,\,-1.05]")
        self.assertEqual(values["DAnatomyTwoRandomEpHundred"], "---")

    def test_replication_differences_without_rescaling_absolute_auc(self):
        bundle = {"numeric": {"RepPlacementDelta": "+0.0070", "RepAUCCentroidSeedA": "0.8721",
                              "RepRandomRange": "0.0012"},
                  "full_precision": {"RepPlacementDelta": .007046286892287812,
                                     "RepAUCCentroidSeedA": .872064313932,
                                     "RepRandomRange": .001234}}
        values = poster.poster_pp_values(bundle)
        self.assertEqual(values["RepPlacementDelta"], "+0.70")
        self.assertEqual(values["RepRandomRange"], "0.12")
        self.assertNotIn("RepAUCCentroidSeedA", values)


if __name__ == "__main__":
    unittest.main()
