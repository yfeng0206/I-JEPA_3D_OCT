"""Frozen ImageNet-subset kNN and linear evaluation.

The weighted kNN implementation follows DINO's cosine-similarity voting
formulation. The linear probe is a shared local cached-feature protocol; it is
not presented as any model's published evaluation recipe.
"""

import hashlib
import warnings

import numpy as np
import torch
import torch.nn.functional as F


def _as_float_tensor(values):
    if isinstance(values, torch.Tensor):
        return values.detach().float().cpu()
    array = np.asarray(values, dtype=np.float32)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array)


def _as_long_tensor(values):
    if isinstance(values, torch.Tensor):
        return values.detach().long().cpu()
    array = np.asarray(values, dtype=np.int64)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array)


def classification_metrics(scores, labels, topk=(1, 5)):
    """Compute top-k and macro top-1 accuracy from class scores."""
    scores = _as_float_tensor(scores)
    labels = _as_long_tensor(labels)
    if scores.dim() != 2:
        raise ValueError("scores must have shape (N, C)")
    if labels.dim() != 1 or labels.numel() != scores.size(0):
        raise ValueError("labels must have shape (N,)")
    if scores.size(1) < 2:
        raise ValueError("classification requires at least two classes")

    results = {}
    maximum = min(max(int(k) for k in topk), scores.size(1))
    predictions = scores.topk(maximum, dim=1).indices
    for k in topk:
        effective = min(int(k), scores.size(1))
        correct = predictions[:, :effective].eq(labels.unsqueeze(1)).any(1)
        results["top%d" % int(k)] = float(correct.float().mean().item())

    top1 = predictions[:, 0]
    class_values = labels.unique(sorted=True)
    per_class = {}
    for class_id in class_values.tolist():
        selected = labels == int(class_id)
        accuracy = top1[selected].eq(labels[selected]).float().mean()
        per_class[int(class_id)] = float(accuracy.item())
    results["macro_top1"] = float(np.mean(list(per_class.values())))
    results["per_class_top1"] = per_class
    return results


def weighted_knn_predict(
    train_features,
    train_labels,
    query_features,
    num_classes,
    k=20,
    temperature=0.07,
    device="cuda",
    train_chunk_size=8192,
    query_chunk_size=256,
):
    """Predict with chunked weighted cosine kNN.

    Features are L2-normalized in FP32. For each query, the top-k cosine
    neighbors vote with ``exp(similarity / temperature)`` weights.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    train_features = F.normalize(_as_float_tensor(train_features), dim=1)
    query_features = F.normalize(_as_float_tensor(query_features), dim=1)
    train_labels = _as_long_tensor(train_labels)
    if train_features.dim() != 2 or query_features.dim() != 2:
        raise ValueError("features must have shape (N, D)")
    if train_features.size(1) != query_features.size(1):
        raise ValueError("train and query feature dimensions differ")
    if train_labels.numel() != train_features.size(0):
        raise ValueError("train label count does not match features")
    if num_classes <= int(train_labels.max().item()):
        raise ValueError("num_classes does not cover train labels")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), train_features.size(0))

    target_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    output_scores = []
    for query_start in range(0, query_features.size(0), query_chunk_size):
        queries = query_features[
            query_start:query_start + query_chunk_size
        ].to(target_device)
        best_values = torch.empty(
            queries.size(0), 0, device=target_device
        )
        best_indices = torch.empty(
            queries.size(0), 0, dtype=torch.long, device=target_device
        )
        for train_start in range(0, train_features.size(0), train_chunk_size):
            train_stop = min(
                train_start + train_chunk_size, train_features.size(0)
            )
            bank = train_features[train_start:train_stop].to(target_device)
            similarities = queries @ bank.t()
            candidate_values = torch.cat(
                [best_values, similarities], dim=1
            )
            local_indices = torch.arange(
                train_start, train_stop, device=target_device
            ).expand(queries.size(0), -1)
            candidate_indices = torch.cat(
                [best_indices, local_indices], dim=1
            )
            keep = min(k, candidate_values.size(1))
            best_values, order = candidate_values.topk(keep, dim=1)
            best_indices = candidate_indices.gather(1, order)

        neighbor_labels = train_labels[best_indices.cpu()].to(target_device)
        weights = torch.exp(best_values / float(temperature))
        scores = torch.zeros(
            queries.size(0),
            int(num_classes),
            dtype=torch.float32,
            device=target_device,
        )
        scores.scatter_add_(1, neighbor_labels, weights)
        output_scores.append(scores.cpu())
    return torch.cat(output_scores, dim=0)


def deterministic_class_split(labels, sample_ids, validation_fraction=0.1):
    """Create a per-class deterministic train/development split.

    Samples are sorted by SHA-256 of their stable sample ID. This split is
    formed entirely from the training set and never consults model output.
    """
    labels = _as_long_tensor(labels)
    if len(sample_ids) != labels.numel():
        raise ValueError("sample ID count does not match labels")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    train_indices = []
    validation_indices = []
    for class_id in labels.unique(sorted=True).tolist():
        indices = torch.nonzero(
            labels == int(class_id), as_tuple=False
        ).flatten().tolist()
        indices.sort(
            key=lambda index: hashlib.sha256(
                str(sample_ids[index]).encode("utf-8")
            ).hexdigest()
        )
        validation_count = max(
            1, int(round(len(indices) * float(validation_fraction)))
        )
        validation_count = min(validation_count, max(1, len(indices) - 1))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    return (
        np.asarray(sorted(train_indices), dtype=np.int64),
        np.asarray(sorted(validation_indices), dtype=np.int64),
    )


def fit_linear_probe(
    train_features,
    train_labels,
    train_sample_ids,
    query_features,
    c_values=None,
    validation_fraction=0.1,
    max_iter=1000,
    tolerance=1e-12,
):
    """Fit a frozen multinomial logistic-regression probe.

    ``C`` is selected on a deterministic per-class split of the training
    features, after which the classifier is refit on the complete train bank.
    """
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("linear probing requires scikit-learn") from exc

    train_features = np.asarray(train_features, dtype=np.float64)
    query_features = np.asarray(query_features, dtype=np.float64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    if train_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if train_features.shape[1] != query_features.shape[1]:
        raise ValueError("train and query feature dimensions differ")
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train labels do not match feature count")
    if c_values is None:
        c_values = np.logspace(-6, 5, 45)
    c_values = [float(value) for value in c_values]
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("C values must be positive")

    fit_indices, development_indices = deterministic_class_split(
        train_labels,
        train_sample_ids,
        validation_fraction=validation_fraction,
    )
    candidates = []
    for value in c_values:
        classifier = LogisticRegression(
            C=value,
            solver="lbfgs",
            max_iter=int(max_iter),
            tol=float(tolerance),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(
                train_features[fit_indices],
                train_labels[fit_indices],
            )
        if any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ):
            raise RuntimeError(
                "linear-probe development fit did not converge for C=%g; "
                "increase max_iter rather than reporting an incomplete fit"
                % value
            )
        development_scores = classifier.decision_function(
            train_features[development_indices]
        )
        if development_scores.ndim == 1:
            development_scores = np.stack(
                [-development_scores, development_scores], axis=1
            )
        accuracy = classification_metrics(
            development_scores,
            train_labels[development_indices],
            topk=(1,),
        )["top1"]
        candidates.append((accuracy, value))

    # Prefer stronger regularization (smaller C) when development scores tie.
    best_accuracy, best_c = max(candidates, key=lambda item: (item[0], -item[1]))
    final_classifier = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=int(max_iter),
        tol=float(tolerance),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final_classifier.fit(train_features, train_labels)
    if any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    ):
        raise RuntimeError(
            "final linear-probe fit did not converge for C=%g; increase "
            "max_iter rather than reporting an incomplete fit" % best_c
        )
    query_scores = final_classifier.decision_function(query_features)
    if query_scores.ndim == 1:
        query_scores = np.stack([-query_scores, query_scores], axis=1)
    return {
        "classifier": final_classifier,
        "scores": query_scores,
        "selected_c": float(best_c),
        "development_top1": float(best_accuracy),
        "candidates": tuple(
            {"c": value, "top1": accuracy}
            for accuracy, value in candidates
        ),
    }
