"""Reproduce row splits and evaluate geographically grouped holdouts."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split

from .features import CLASS_NAMES


def new_forest(seed=42):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  class_weight="balanced_subsample", random_state=seed,
                                  n_jobs=-1)


def score_predictions(y, predicted):
    labels = list(CLASS_NAMES)
    return {"n_test": len(y), "accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro",
                                       zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
            "per_class": classification_report(y, predicted, labels=labels,
                                                output_dict=True, zero_division=0)}


def spatial_groups(xy, block_size):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Finite projected x/y coordinate pairs are required")
    if not np.isfinite(block_size) or block_size <= 0:
        raise ValueError("Block size must be a positive distance in CRS metres")
    cells = np.floor(xy / block_size).astype(np.int64)
    return np.array([f"{x}:{y}" for x, y in cells])


def buffered_training_indices(train, test, xy, buffer_m):
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("Buffer must be a finite nonnegative distance")
    if buffer_m == 0:
        return train
    distances, _ = cKDTree(np.asarray(xy)[test]).query(np.asarray(xy)[train])
    return train[distances > buffer_m]


def evaluate(X, y, xy=None, block_size=1000, buffer_m=300, n_splits=5, seed=42):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    if X.ndim != 2 or len(X) != len(y) or len(y) < 10 or not np.isfinite(X).all():
        raise ValueError("Need at least ten aligned, finite samples")
    if xy is not None and len(xy) != len(y):
        raise ValueError("Coordinates and samples must align")
    idx = np.arange(len(y))
    counts = np.unique(y, return_counts=True)[1]
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                              stratify=y if counts.min() >= 2 else None)
    splits = [("random", 0, tr, te)]
    groups = None
    if xy is not None:
        groups = spatial_groups(xy, block_size)
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError("At least two distinct spatial groups required")
        kfold = GroupKFold(n_splits=min(n_splits, n_groups))
        for fold, (tr, te) in enumerate(kfold.split(X, y, groups)):
            if set(groups[tr]) & set(groups[te]):
                raise AssertionError("Spatial group leakage")
            tr = buffered_training_indices(tr, te, xy, buffer_m)
            if not len(tr):
                raise ValueError("Buffer removes all training samples; revise sampling design")
            splits.append(("spatial_buffered", fold, tr, te))
    reports, predictions, assignments = [], [], []
    for method, fold, train, test in splits:
        for model_name, estimator in [("majority", DummyClassifier(strategy="most_frequent")),
                                       ("random_forest", new_forest(seed))]:
            model = clone(estimator).fit(X[train], y[train])
            predicted = model.predict(X[test])
            reports.append({"split": method, "fold": fold, "model": model_name,
                            "n_train": len(train),
                            "missing_training_classes": sorted(set(CLASS_NAMES) - set(y[train])),
                            **score_predictions(y[test], predicted)})
            for i, pred in zip(test, predicted):
                predictions.append({"split": method, "fold": fold, "model": model_name,
                                    "sample_index": int(i), "actual": int(y[i]),
                                    "predicted": int(pred),
                                    "group": str(groups[i]) if groups is not None else "unknown"})
        for role, indices in [("train", train), ("test", test)]:
            assignments += [{"split": method, "fold": fold, "role": role,
                             "sample_index": int(i),
                             "group": str(groups[i]) if groups is not None else "unknown"}
                            for i in indices]
    summary = []
    frame = pd.DataFrame(predictions)
    for (method, model), group in frame.groupby(["split", "model"]):
        detail = [r for r in reports if r["split"] == method and r["model"] == model]
        summary.append({"split": method, "model": model,
                        "fold_macro_f1_mean": float(np.mean([r["macro_f1"] for r in detail])),
                        "fold_macro_f1_std": float(np.std([r["macro_f1"] for r in detail])),
                        **score_predictions(group.actual, group.predicted)})
    return {"n_samples": len(y), "n_groups": len(np.unique(groups)) if groups is not None else None,
            "block_size_m": block_size if groups is not None else None,
            "buffer_m": buffer_m if groups is not None else None,
            "duplicate_feature_rows": int(pd.DataFrame(X).duplicated(keep=False).sum()),
            "summaries": summary, "folds": reports}, frame, pd.DataFrame(assignments)
