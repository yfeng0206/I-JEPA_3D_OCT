import unittest

from src.evaluation.grounding import (
    box_iou,
    evaluate_box_record,
    evaluate_point_record,
    grounding_box_pixels,
)


class GroundingMetricTests(unittest.TestCase):
    def test_box_iou(self):
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (10, 10, 20, 20)), 0.0)
        self.assertAlmostEqual(
            box_iou((0, 0, 10, 10), (5, 5, 15, 15)),
            25.0 / 175.0,
        )

    def test_normalized_box_scales_to_annotation_image(self):
        box = {
            "bbox_2d": [100, 200, 900, 800],
            "coordinate_space": "normalized_1000",
        }
        self.assertEqual(
            grounding_box_pixels(box, (200, 100)),
            (20.0, 20.0, 180.0, 80.0),
        )

    def test_box_record_uses_best_pair(self):
        record = {
            "grounding_regions": [
                {
                    "label": "dog",
                    "bbox_2d": [100, 100, 500, 500],
                    "coordinate_space": "normalized_1000",
                },
                {
                    "label": "ball",
                    "bbox_2d": [500, 500, 900, 900],
                    "coordinate_space": "normalized_1000",
                },
            ]
        }
        annotation = {
            "image_size": [100, 100],
            "boxes": [{"bbox_xyxy": [50, 50, 90, 90]}],
        }
        result = evaluate_box_record(record, annotation)
        self.assertAlmostEqual(result["best_iou"], 1.0)
        self.assertEqual(result["best"]["prediction_label"], "ball")

    def test_point_record_reports_any_truth_hit(self):
        record = {
            "grounding_points": [
                {
                    "label": "dog",
                    "point_2d": [500, 500],
                    "coordinate_space": "normalized_1000",
                }
            ]
        }
        annotation = {
            "image_size": [200, 100],
            "boxes": [{"bbox_xyxy": [80, 40, 120, 60]}],
        }
        result = evaluate_point_record(record, annotation)
        self.assertTrue(result["any_point_hit"])
        self.assertEqual(result["point_hit_count"], 1)


if __name__ == "__main__":
    unittest.main()
