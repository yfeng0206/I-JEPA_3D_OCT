import copy
import json
import unittest

import numpy as np
from PIL import Image

import audit_legacy_figures as audit


class LegacyReviewTests(unittest.TestCase):
    def test_affine_checks_every_tick(self):
        mapping = audit.affine([0, 1, 2], [5, 15, 25])
        self.assertAlmostEqual(audit.decode(20, mapping), 1.5)
        with self.assertRaises(ValueError):
            audit.affine([0, 1, 2], [5, 15, 26])

    def test_duplicate_sources_are_rejected_even_when_identical(self):
        receipt = {"version": 1, "sources": {"same": {"path": "same"}}}
        with self.assertRaisesRegex(ValueError, "Duplicate sources"):
            audit.merge_reviews(receipt, receipt)

    def test_duplicate_figure_path_separators_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate figures"):
            audit.merge_reviews({"version": 1, "figures": [{"path": r"figures\plot.png"}]},
                                {"version": 1, "figures": [{"path": "figures/plot.png"}]})

    def test_duplicate_literal_identity_is_rejected(self):
        row = {"file": "main.tex", "context_sha256": "a" * 64, "token_index": 1}
        with self.assertRaisesRegex(ValueError, "Duplicate literals"):
            audit.merge_reviews({"version": 1, "literals": [row]},
                                {"version": 1, "literals": [row]})

    def test_merge_preserves_inputs(self):
        base = {"version": 1, "sources": {"a": {"path": "a"}}, "figures": []}
        candidate = {"version": 1, "sources": {"b": {"path": "b"}}, "figures": []}
        before = copy.deepcopy(base)
        result = audit.merge_reviews(base, candidate)
        self.assertEqual(base, before)
        self.assertEqual(set(result["sources"]), {"a", "b"})
        self.assertEqual(result["figures"], [])

    def test_real_precision_figure_is_not_silently_approved(self):
        result = audit.precision_audit()
        self.assertEqual(len(result["points"]), 5)
        self.assertTrue(result["png_geometry"]["all_marker_centers_concordant"])
        self.assertTrue(all(p["auc_matches_rounding_to_6_decimals"] for p in result["points"]))
        self.assertTrue(all(not p["purity_matches_rounding_to_1_decimal"] for p in result["points"]))
        self.assertLess(result["regression"]["max_abs_pdf_line_residual_auc"], 1e-8)

    def test_altered_raster_marker_fails_geometry_check(self):
        result = audit.precision_audit()
        rgb = np.asarray(Image.open(audit.PAPER / "figures" / "fig_precision_paradox.png").convert("RGB")).copy()
        original = rgb[245:298, 393:445].copy()
        rgb[245:298, 393:445] = 255
        rgb[245:298, 413:465] = original
        inspected = audit.raster_marks(Image.fromarray(rgb), result["points"], result["pdf_axis_bounds"])
        self.assertFalse(inspected["all_marker_centers_concordant"])

    def test_candidate_has_no_empirical_or_illustration_approvals(self):
        candidate = json.loads((audit.HERE / "candidate_numeric_reviews.json").read_text(encoding="utf-8"))
        import jsonschema
        schema = json.loads((audit.REPO / "autopilot" / "numeric_reviews.schema.json").read_text())
        jsonschema.validate(candidate, schema)
        self.assertEqual(candidate["figures"], [])
        for spec in candidate["sources"].values():
            root = {"repo": audit.REPO, "paper": audit.PAPER, "stats": audit.STATS}[spec["root"]]
            self.assertEqual(audit.sha(root / spec["path"]), spec["sha256"])

    def test_report_caption_and_input_bindings_are_current(self):
        from autopilot.numeric_bindings import Evidence
        report = json.loads((audit.HERE / "inspection.json").read_text(encoding="utf-8"))
        evidence = Evidence(audit.PAPER, audit.STATS, report["sources"])
        manuscript = (audit.PAPER / "main_submission.tex").read_text(encoding="utf-8")
        for asset in report["assets"]:
            caption = audit.figure_context(manuscript, asset["path"])
            self.assertEqual(audit.digest_text(caption), asset["caption_sha256"])
            self.assertEqual(audit.sha(audit.PAPER / asset["path"]), asset["sha256"])
            self.assertTrue(asset["rationale"])
            self.assertTrue(asset["limitations"])
            for reference in asset["inputs"]:
                evidence.evaluate(reference)
                self.assertEqual(evidence.hashes[reference["source"]], reference["sha256"])

    def test_mask_headers_match_source_but_are_not_mask_verification(self):
        result = audit.mask_annotation_checks()
        self.assertEqual(len(result), 6)
        self.assertTrue(all(row["label_matches"] for row in result))
        self.assertTrue(all("not a mask-pixel" in row["scope"] for row in result))


if __name__ == "__main__":
    unittest.main()
