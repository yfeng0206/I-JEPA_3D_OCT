import unittest

import numpy as np
import torch

from scripts.phase0_imagenet50_eval import feature_guide_identity
from src.evaluation.imagenet_frozen import (
    classification_metrics,
    deterministic_class_split,
    fit_linear_probe,
    weighted_knn_predict,
)


class FrozenEvaluationTests(unittest.TestCase):
    def test_weighted_knn_predicts_cluster_labels(self):
        train_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        train_labels = torch.tensor([0, 0, 1, 1])
        query_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        scores = weighted_knn_predict(
            train_features,
            train_labels,
            query_features,
            num_classes=2,
            k=2,
            device="cpu",
            train_chunk_size=2,
            query_chunk_size=1,
        )
        self.assertEqual(scores.argmax(dim=1).tolist(), [0, 1])

    def test_classification_metrics_include_macro_accuracy(self):
        scores = torch.tensor(
            [[3.0, 1.0], [0.0, 2.0], [0.0, 2.0]]
        )
        labels = torch.tensor([0, 1, 0])
        metrics = classification_metrics(scores, labels, topk=(1, 5))
        self.assertAlmostEqual(metrics["top1"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["top5"], 1.0, places=6)
        self.assertAlmostEqual(metrics["macro_top1"], 0.75, places=6)

    def test_deterministic_split_is_class_balanced_and_stable(self):
        labels = np.asarray([0] * 10 + [1] * 10)
        sample_ids = ["sample-%02d" % index for index in range(20)]
        first = deterministic_class_split(labels, sample_ids, 0.2)
        second = deterministic_class_split(labels, sample_ids, 0.2)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        development_labels = labels[first[1]]
        self.assertEqual(int((development_labels == 0).sum()), 2)
        self.assertEqual(int((development_labels == 1).sum()), 2)

    def test_linear_probe_selects_and_predicts(self):
        rng = np.random.default_rng(11)
        first = rng.normal(loc=-2.0, scale=0.2, size=(20, 4))
        second = rng.normal(loc=2.0, scale=0.2, size=(20, 4))
        features = np.concatenate([first, second], axis=0)
        labels = np.asarray([0] * 20 + [1] * 20)
        sample_ids = ["sample-%02d" % index for index in range(40)]
        result = fit_linear_probe(
            features,
            labels,
            sample_ids,
            features,
            c_values=(0.1, 1.0),
            max_iter=200,
            tolerance=1e-8,
        )
        metrics = classification_metrics(result["scores"], labels)
        self.assertGreaterEqual(metrics["top1"], 0.99)
        self.assertIn(result["selected_c"], (0.1, 1.0))

    def test_multiclass_score_columns_follow_contiguous_labels(self):
        rng = np.random.default_rng(19)
        features = np.concatenate(
            [
                rng.normal(-3.0, 0.1, size=(12, 3)),
                rng.normal(0.0, 0.1, size=(12, 3)),
                rng.normal(3.0, 0.1, size=(12, 3)),
            ],
            axis=0,
        )
        labels = np.repeat(np.arange(3), 12)
        sample_ids = ["sample-%02d" % index for index in range(36)]
        result = fit_linear_probe(
            features,
            labels,
            sample_ids,
            features,
            c_values=(1.0,),
            max_iter=200,
            tolerance=1e-8,
        )
        self.assertEqual(result["classifier"].classes_.tolist(), [0, 1, 2])
        self.assertEqual(result["scores"].shape, (36, 3))
        self.assertGreater(
            classification_metrics(result["scores"], labels)["top1"],
            0.99,
        )

    def test_feature_identity_ignores_probe_and_atlas_runtime_fields(self):
        common = {
            "model_id": "official/model",
            "revision": "abc",
            "input_size": 224,
            "dtype": "float32",
        }
        left = {
            **common,
            "atlas_batch_size": 1,
            "evaluation_batch_size": 8,
            "caption_max_new_tokens": 32,
        }
        right = {
            **common,
            "atlas_batch_size": 4,
            "evaluation_batch_size": 64,
            "caption_max_new_tokens": 96,
        }
        self.assertEqual(
            feature_guide_identity(left),
            feature_guide_identity(right),
        )
        changed = {**right, "revision": "def"}
        self.assertNotEqual(
            feature_guide_identity(left),
            feature_guide_identity(changed),
        )


if __name__ == "__main__":
    unittest.main()
