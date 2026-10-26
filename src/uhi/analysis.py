"""Masked prediction, paired ward summaries, and exploratory associations."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .features import feature_rows, restore_labels


def predict_masked(model, cube, valid, batch_size=65536):
    if batch_size <= 0:
        raise ValueError("Prediction batch size must be positive")
    rows = feature_rows(cube, valid)
    labels = np.empty(len(rows), dtype=np.uint8)
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        labels[start:stop] = model.predict(rows[start:stop])
    return restore_labels(labels, valid)


def paired_ward_statistics(wards, zones, first, second, classes_first, classes_second,
                           ndvi_first, ndvi_second):
    arrays = [first, second, classes_first, classes_second, ndvi_first, ndvi_second]
    if any(a.shape != zones.shape for a in arrays):
        raise ValueError("Every array must share the ward grid")
    paired = np.isfinite(first) & np.isfinite(second) & (zones > 0)
    paired_classes = (classes_first > 0) & (classes_second > 0) & (zones > 0)
    paired_ndvi = paired & np.isfinite(ndvi_first) & np.isfinite(ndvi_second)
    rows = []
    for i, ward in wards.reset_index(drop=True).iterrows():
        region = zones == i + 1
        mask, cover, vegetation = region & paired, region & paired_classes, region & paired_ndvi
        row = {"ward_id": ward.ward_id, "ward_name": ward.ward_name,
               "total_grid_pixels": int(region.sum()), "paired_lst_pixels": int(mask.sum()),
               "paired_lst_fraction": float(mask.sum() / region.sum()) if region.any() else 0,
               "paired_class_pixels": int(cover.sum()), "paired_ndvi_pixels": int(vegetation.sum())}
        for year, values in [(2016, first), (2022, second)]:
            a = values[mask]
            for key, fn in [("mean", np.mean), ("median", np.median),
                            ("p10", lambda x: np.percentile(x, 10)),
                            ("p90", lambda x: np.percentile(x, 90))]:
                row[f"lst_{year}_{key}_c"] = float(fn(a)) if len(a) else np.nan
        delta = second[mask] - first[mask]
        row["lst_delta_mean_c"] = float(np.mean(delta)) if len(delta) else np.nan
        row["lst_delta_median_c"] = float(np.median(delta)) if len(delta) else np.nan
        row["ndvi_delta_mean"] = float(np.mean(ndvi_second[vegetation] - ndvi_first[vegetation])) if vegetation.any() else np.nan
        if cover.any():
            old, new = classes_first[cover], classes_second[cover]
            row["built_fraction_2016"] = float(np.isin(old, [4, 5]).mean())
            row["built_fraction_2022"] = float(np.isin(new, [4, 5]).mean())
            row["built_fraction_delta"] = row["built_fraction_2022"] - row["built_fraction_2016"]
            row["bare_to_built_fraction"] = float(((old == 3) & np.isin(new, [4, 5])).mean())
        else:
            for name in ("built_fraction_2016", "built_fraction_2022", "built_fraction_delta", "bare_to_built_fraction"):
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows), paired, paired_classes


def describe(values):
    values = np.asarray(values)
    a = values[np.isfinite(values)]
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def transition_table(first, second, valid, pixel_area_m2):
    if first.shape != second.shape or first.shape != valid.shape:
        raise ValueError("Transition arrays must share a grid")
    if not np.isin(first[valid], range(1, 6)).all() or not np.isin(second[valid], range(1, 6)).all():
        raise ValueError("A transition cannot include NoData or an unknown class")
    rows = []
    for old in range(1, 6):
        for new in range(1, 6):
            n = int((valid & (first == old) & (second == new)).sum())
            rows.append({"class_2016": old, "class_2022": new, "pixels": n,
                         "map_grid_area_km2": n * pixel_area_m2 / 1e6})
    return pd.DataFrame(rows)


def exploratory_correlations(frame, min_coverage=.8):
    # Omit naive IID p-values: neighboring wards can have correlated errors.
    rows = []
    for name in ("ndvi_delta_mean", "built_fraction_delta", "bare_to_built_fraction"):
        data = frame.loc[frame.paired_lst_fraction >= min_coverage, [name, "lst_delta_mean_c"]].dropna()
        r = float(pearsonr(data[name], data.lst_delta_mean_c).statistic) if len(data) >= 3 and data[name].nunique() > 1 and data.lst_delta_mean_c.nunique() > 1 else np.nan
        rows.append({"x": name, "y": "lst_delta_mean_c", "n_wards": len(data), "pearson_r": r,
                     "minimum_paired_lst_fraction": min_coverage,
                     "interpretation": "Exploratory association; spatial dependence, sampling and scene differences preclude causal attribution"})
    return rows
