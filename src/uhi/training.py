"""Recover source-label provenance and sample a shared optical feature contract."""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .features import FEATURE_NAMES, optical_features, quality_mask, validate_training
from .evaluation import score_predictions
from .geospatial import scene_paths

SOURCE_FILES = {
    2016: ("Training_2016_Fixed1.shp", "training_2016_fixed1.csv"),
    2022: ("Training_Fixed_1.shp", "training_2022_fixed.csv"),
}


def sample_training(root, year):
    shape, csv = SOURCE_FILES[year]
    points = gpd.read_file(root / "processed" / shape)
    original = pd.read_csv(root / "processed" / csv)
    if points.crs is None or not points.geom_type.eq("Point").all():
        raise ValueError("Labeled points with a known CRS are required")
    if len(points) != len(original) or not np.array_equal(
            points["class"], original["class"], equal_nan=True):
        raise ValueError("Source CSV and point-label row order differ")
    paths = scene_paths(root, year)
    with rasterio.open(paths["SR_B2"]) as src:
        points = points.to_crs(src.crs)
    xy = np.c_[points.geometry.x, points.geometry.y]

    def sample(name):
        with rasterio.open(paths[name]) as src:
            if src.crs != points.crs:
                raise ValueError("Sample source CRS mismatch")
            return np.array(list(src.sample(xy)))[:, 0]

    raw = np.array([sample(f"SR_B{b}") for b in (2, 3, 4, 5, 6)])
    expected = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ST_B10"] if year == 2016 else ["SR_B4", "SR_B3", "SR_B2"]
    # Exact DN equality establishes which physical bands the old SAMPLE columns contain.
    mapping = {}
    for i, band in enumerate(expected, 1):
        name = f"SAMPLE_{i}"
        matches = np.array_equal(sample(band), original[name].to_numpy())
        if not matches:
            raise ValueError(f"Point coordinates do not reproduce original {name}")
        mapping[name] = band
    features = optical_features(raw).T
    valid = quality_mask(sample("QA_PIXEL"), sample("QA_RADSAT")) & (raw > 0).all(axis=0)
    frame = pd.DataFrame(features, columns=FEATURE_NAMES)
    frame.insert(0, "sample_id", [f"{year}-{i:03d}" for i in range(len(frame))])
    frame["source_row"] = np.arange(len(frame))
    frame["class"] = points["class"].to_numpy()
    frame["x_m"], frame["y_m"] = xy.T
    frame["qa_valid"] = valid
    clean = validate_training(frame[valid], FEATURE_NAMES).reset_index(drop=True)
    clean["class"] = clean["class"].astype(int)
    provenance = {"year": year, "source_points": len(points), "retained_points": len(clean),
                  "missing_labels": int(points["class"].isna().sum()),
                  "qa_rejected": int((~valid).sum()), "crs": points.crs.to_string(),
                  "csv_band_mapping_verified_exactly": mapping,
                  "class_counts": clean["class"].value_counts().sort_index().to_dict(),
                  "duplicate_coordinates": int(clean.duplicated(["x_m", "y_m"], keep=False).sum())}
    return clean, provenance


def legacy_baseline(root, year):
    """Original 100-tree unstratified 80/20 recipe, fixing required-column cleanup only."""
    _, name = SOURCE_FILES[year]
    frame = pd.read_csv(root / "processed" / name)
    names = [c for c in frame if c.startswith("SAMPLE_")]
    clean = validate_training(frame, names)
    X, y = clean[names].to_numpy(), clean["class"].astype(int).to_numpy()
    train, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pred = model.fit(X[train], y[train]).predict(X[test])
    return {"rows_before": len(frame), "rows_after_required_cleanup": len(clean),
            "rows_after_old_blanket_dropna": len(frame.dropna()),
            "n_features": len(names), "split": "unstratified_random_80_20_seed42",
            "warning": "Row holdout on manually sampled points; not independent map accuracy",
            **score_predictions(y[test], pred)}
