import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.transforms import IdentityTransform
import numpy as np

import generate_replacements as producer
import verify_replacements as verifier


class ReplacementTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def maps(self):
        return producer.build_token_maps(producer.frozen_token_data()[0])

    def test_real_scatter_sources_and_exact_png(self):
        fig = producer.build_scatter()
        evidence = verifier.verify_scatter_artists(fig)
        self.assertEqual(len(evidence["series"]), 5)
        self.assertEqual({r["precision"] for r in evidence["series"]}, {"fp32"})
        self.assertTrue(verifier.verify_png(fig, producer.OUT / (producer.SCATTER_STEM + ".png"))["exact_byte_equality"])
        verifier.verify_text_layout(fig)

    def test_real_token_arrays_and_exact_png(self):
        fig = self.maps()
        evidence = verifier.verify_token_artists(fig)
        self.assertEqual(sum(r["cells_checked"] for r in evidence["policies"]), 1280)
        self.assertEqual(evidence["raw_image_artists"], 0)
        self.assertTrue(verifier.verify_png(fig, producer.OUT / (producer.MAP_STEM + ".png"))["exact_byte_equality"])
        verifier.verify_text_layout(fig)

    def test_canonical_random_label_in_figures_and_registration(self):
        scatter = producer.build_scatter()
        maps = self.maps()
        self.assertEqual(scatter.axes[0].texts[0].get_text(), "Random")
        self.assertEqual(maps.axes[0].get_title(), "(a) Random")
        registration = json.loads((producer.OUT / "numeric_validator_registration.json").read_text())
        baseline = next(row for row in registration["expected_semantic_series"] if row["arm"] == "random")
        self.assertEqual(baseline["display_label"], "Random")

    def test_fp16_value_is_not_accepted_for_fp32_point(self):
        fig = producer.build_scatter()
        inventory = json.loads(producer.INVENTORY.read_text())
        fp16 = next(r for r in inventory["records"] if r.get("arm") == "random"
                    and r.get("epoch") == 50 and r.get("precision") == "fp16")
        values = fig.axes[0].collections[0].get_offsets().copy()
        values[0, 1] = fp16["auc"]
        fig.axes[0].collections[0].set_offsets(values)
        with self.assertRaisesRegex(ValueError, "exact fp32 point"):
            verifier.verify_scatter_artists(fig)

    def test_duplicate_inventory_rows_rejected(self):
        fig = producer.build_scatter()
        original = Path.read_text
        inventory = json.loads(original(producer.INVENTORY))
        row = next(r for r in inventory["records"] if r.get("arm") == "random"
                   and r.get("epoch") == 50 and r.get("precision") == "fp32")
        inventory["records"].append(copy.deepcopy(row))

        def replacement(path, *args, **kwargs):
            return json.dumps(inventory) if path == producer.INVENTORY else original(path, *args, **kwargs)

        with patch.object(Path, "read_text", replacement):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                verifier.verify_scatter_artists(fig)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                producer.build_scatter()

    def test_fitted_line_rejected(self):
        fig = producer.build_scatter()
        fig.axes[0].plot([30, 90], [.864, .875])
        with self.assertRaisesRegex(ValueError, "Fitted/reference"):
            verifier.verify_scatter_artists(fig)

    def test_changed_scatter_transform_rejected(self):
        fig = producer.build_scatter()
        fig.axes[0].collections[0].set_offset_transform(IdentityTransform())
        with self.assertRaisesRegex(ValueError, "transform changed"):
            verifier.verify_scatter_artists(fig)

    def test_nonmatching_png_replay_rejected(self):
        fig = producer.build_scatter()
        fig.axes[0].collections[0].set_offsets([[33, .865]])
        with self.assertRaisesRegex(ValueError, "PNG differs"):
            verifier.verify_png(fig, producer.OUT / (producer.SCATTER_STEM + ".png"))

    def test_frozen_inputs_are_read_without_seed_redraw(self):
        import torch
        with patch.object(torch, "manual_seed", side_effect=AssertionError("seed redraw")), \
                patch.object(np.random, "seed", side_effect=AssertionError("seed redraw")):
            tokens, metadata = producer.frozen_token_data()
        self.assertEqual(metadata["anonymous_ordinal"], 0)
        self.assertEqual(metadata["private_fixture_sha256"], producer.FIXTURE_SHA256)
        self.assertEqual([t["fixture_arm"] for t in tokens],
                         ["random", "oracle", "envelope", "anatomy", "cover_legacy"])

    def test_wrong_token_membership_rejected(self):
        fig = self.maps()
        cell = fig.axes[0].patches[0]
        cell.set_facecolor("#E69F00")
        with self.assertRaisesRegex(ValueError, "source token class"):
            verifier.verify_token_artists(fig)

    def test_missing_guide_circle_rejected(self):
        fig = self.maps()
        circles = fig.axes[0].collections[0]
        circles.set_offsets(circles.get_offsets()[1:])
        with self.assertRaisesRegex(ValueError, "guide-positive circles"):
            verifier.verify_token_artists(fig)

    def test_hidden_raster_image_rejected(self):
        fig = self.maps()
        fig.axes[0].imshow(np.zeros((2, 2)), visible=False)
        with self.assertRaisesRegex(ValueError, "Raw/raster"):
            verifier.verify_token_artists(fig)

    def test_hidden_clinical_text_rejected(self):
        fig = self.maps()
        fig.text(0, 0, "pred=0.99; RNFL pathology", visible=False)
        with self.assertRaisesRegex(ValueError, "clinical annotations"):
            verifier.verify_token_artists(fig)

    def test_false_anatomy_v1_label_rejected(self):
        fig = self.maps()
        fig.axes[3].set_title("(d) ANATOMY-v1")
        with self.assertRaisesRegex(ValueError, "token-map text"):
            verifier.verify_token_artists(fig)

    def test_misleading_legend_color_rejected(self):
        fig = self.maps()
        fig.legends[0].legend_handles[0].set_facecolor("#56B4E9")
        with self.assertRaisesRegex(ValueError, "legend target"):
            verifier.verify_token_artists(fig)

    def test_vectors_have_no_raster_exports(self):
        result = verifier.verify_vectors(producer.OUT)
        self.assertEqual(set(result), {producer.SCATTER_STEM, producer.MAP_STEM})
        self.assertTrue(all(r["pdf_embedded_images"] == r["svg_image_elements"] == 0 for r in result.values()))

    def test_public_manifest_contains_no_raw_arrays_or_case_identifiers(self):
        manifest = json.loads((producer.OUT / "source_manifest.json").read_text())
        tokens = manifest["token_maps"]
        self.assertFalse(tokens["private_fixture_redistributed"])
        self.assertFalse(tokens["raw_oct_pixels_rendered"])
        self.assertFalse(tokens["per_case_identifiers_exported"])
        forbidden = {"images", "guides", "image", "patient_id", "subject_id",
                     "tissue_labels", "target_union", "masks_enc", "masks_pred"}

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
        inspect(tokens)
        for stem, outputs in manifest["outputs"].items():
            for item in outputs.values():
                self.assertEqual(producer.sha(producer.OUT / item["path"]), item["sha256"])

    def test_scatter_expressions_resolve_exact_source_fields(self):
        sys.path.insert(0, str(producer.ROOT))
        from autopilot.numeric_bindings import Evidence
        manifest = json.loads((producer.OUT / "source_manifest.json").read_text())
        evidence = Evidence(producer.ROOT / "paper" / "genai4health2026",
                            producer.INVENTORY.parent, manifest["public_numeric_sources"])
        for row in manifest["scatter"]["series"]:
            self.assertEqual(evidence.evaluate(row["x_expression"]), row["x"])
            self.assertEqual(evidence.evaluate(row["y_expression"]), row["y"])

    def test_source_illustration_receipt_real_api_and_caption_hash(self):
        sys.path.insert(0, str(producer.ROOT))
        from autopilot.numeric_bindings import Evidence, figure_receipt, read_reviews
        import jsonschema
        path = producer.OUT / "candidate_replacement_reviews.json"
        candidate = json.loads(path.read_text())
        schema = json.loads((producer.ROOT / "autopilot" / "numeric_reviews.schema.json").read_text())
        jsonschema.validate(candidate, schema)
        evidence = Evidence(producer.ROOT / "paper" / "genai4health2026", producer.INVENTORY.parent)
        loaded = read_reviews(evidence.roots["paper"], evidence, path)
        self.assertEqual(len(loaded["figures"]), 1)
        entry = loaded["figures"][0]
        item = {"path": entry["path"], "sha256": producer.sha(producer.OUT / (producer.MAP_STEM + ".png"))}
        result = figure_receipt(item, entry, evidence, producer.MAP_CAPTION)
        self.assertEqual(result["status"], "reviewed_source_illustration")
        self.assertFalse(result["mathematically_verified"])
        with self.assertRaisesRegex(ValueError, "caption changed"):
            figure_receipt(item, entry, evidence, producer.MAP_CAPTION + " Different claim.")


if __name__ == "__main__":
    unittest.main()
