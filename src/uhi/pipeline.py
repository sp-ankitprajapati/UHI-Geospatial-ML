"""A single auditable local run, with scientific limitations in the result itself."""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import rasterio

from .analysis import (describe, exploratory_correlations, paired_ward_statistics,
                       predict_masked, transition_table)
from .evaluation import evaluate, new_forest
from .features import CLASS_NAMES, FEATURE_NAMES
from .geospatial import (grid_contract, load_wards, read_scene, reference_grid,
                         scene_paths, write_raster)
from .training import SOURCE_FILES, legacy_baseline, sample_training


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_results(out, wards, profile, scenes, paired, evaluations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.plot import plotting_extent
    extent = plotting_extent(scenes[2016]["temperature"], profile["transform"])
    values = np.r_[scenes[2016]["temperature"][paired], scenes[2022]["temperature"][paired]]
    low, high = np.percentile(values, [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    images = [np.where(paired, scenes[y]["temperature"], np.nan) for y in (2016, 2022)]
    images.append(images[1] - images[0])
    delta_limit = max(abs(np.nanpercentile(images[2], [2, 98])))
    for i, (ax, arr, title) in enumerate(zip(axes, images, ["23 May 2016", "24 May 2022", "2022 minus 2016"])):
        artist = ax.imshow(arr, extent=extent, origin="upper", cmap="inferno" if i < 2 else "coolwarm",
                           vmin=low if i < 2 else -delta_limit, vmax=high if i < 2 else delta_limit)
        wards.boundary.plot(ax=ax, linewidth=.22, color="#607080", alpha=.65)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Easting (m), EPSG:32644")
        if i == 0:
            ax.set_ylabel("Northing (m)")
        fig.colorbar(artist, ax=ax, shrink=.72, label="LST (°C)" if i < 2 else "Difference (°C)")
        ax.tick_params(labelsize=8)
    fig.suptitle("Hyderabad supplied wards: paired clear pixels from two Landsat observations", fontsize=16)
    fig.savefig(out / "lst_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    positions = np.arange(2)
    for j, method in enumerate(("random", "spatial_buffered")):
        values = [next(r["accuracy"] for r in evaluations[y]["summaries"]
                       if r["split"] == method and r["model"] == "random_forest") for y in (2016, 2022)]
        bars = ax.bar(positions + (j - .5) * .33, values, width=.33,
                     label="Random row holdout" if j == 0 else "Spatial blocks + 300 m buffer",
                     color="#227c9d" if j == 0 else "#e07736")
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4)
    ax.set_xticks(positions, ["2016: 157 retained points", "2022: 161 retained points"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Random splits conceal poor transfer across sampled locations")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False)
    fig.savefig(out / "validation_comparison.png", dpi=180)
    plt.close(fig)


def run(data_root, output, block_size=1000, buffer_m=300, plots=True):
    root, out = Path(data_root).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raster_dir, model_dir = out / "rasters", out / "models"
    raster_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    ward_path = root / "shapefiles/Hyderabad/ghmc-wards.geojson"
    reference_path = scene_paths(root, 2016)["SR_B2"]
    with rasterio.open(reference_path) as ref:
        wards, repairs = load_wards(ward_path, ref.crs)
    profile, zones = reference_grid(reference_path, wards)
    wards.to_file(out / "wards.geojson", driver="GeoJSON")
    print(f"Analysis grid {profile['width']} x {profile['height']}; {len(wards)} ward features", flush=True)
    evaluations, provenance, legacy, scenes, classes, feature_importance = {}, {}, {}, {}, {}, []
    used_inputs = [ward_path]
    for year in (2016, 2022):
        print(f"Sampling and evaluating {year}", flush=True)
        frame, provenance[year] = sample_training(root, year)
        frame.to_csv(out / f"training_samples_{year}.csv", index=False)
        X, y = frame[list(FEATURE_NAMES)].to_numpy(), frame["class"].to_numpy()
        xy = frame[["x_m", "y_m"]].to_numpy()
        evaluation, predictions, assignments = evaluate(X, y, xy, block_size, buffer_m)
        evaluations[year] = evaluation
        predictions.to_csv(out / f"predictions_{year}.csv", index=False)
        assignments.to_csv(out / f"split_assignments_{year}.csv", index=False)
        legacy[year] = legacy_baseline(root, year)
        model = new_forest().fit(X, y)
        joblib.dump(model, model_dir / f"experimental_lulc_{year}.joblib")
        write_json(model_dir / f"experimental_lulc_{year}.json", {
            "features": FEATURE_NAMES, "class_names": CLASS_NAMES, "trained_classes": model.classes_.tolist(),
            "status": "experimental_only_poor_spatial_validation", "random_state": 42,
            "model_parameters": model.get_params(), "training_file_sha256": sha256(out / f"training_samples_{year}.csv")})
        feature_importance += [{"year": year, "feature": name, "impurity_importance": float(weight),
                                "caution": "Training impurity importance is biased and does not establish temperature drivers"}
                               for name, weight in zip(FEATURE_NAMES, model.feature_importances_)]
        print(f"Reading and aligning {year} bands", flush=True)
        scene = read_scene(root, year, profile, zones > 0)
        scenes[year] = scene
        classes[year] = predict_masked(model, scene["features"], scene["optical_valid"])
        write_raster(raster_dir / f"lst_{year}_c.tif", scene["temperature"], profile)
        write_raster(raster_dir / f"st_uncertainty_{year}_k.tif", scene["st_uncertainty_k"], profile)
        write_raster(raster_dir / f"experimental_lulc_{year}.tif", classes[year], profile, labels=True)
        for index, name in [(5, "ndvi"), (6, "ndbi")]:
            write_raster(raster_dir / f"{name}_{year}.tif", np.where(scene["optical_valid"], scene["features"][index], np.nan), profile)
        used_inputs.extend(scene_paths(root, year).values())
        shape, csv = SOURCE_FILES[year]
        used_inputs.extend(root / "processed" / (Path(shape).stem + suffix) for suffix in (".shp", ".shx", ".dbf", ".prj"))
        used_inputs.append(root / "processed" / csv)
    pd.DataFrame(feature_importance).to_csv(out / "experimental_feature_importance.csv", index=False)
    print("Aggregating paired pixels by ward", flush=True)
    table, paired, paired_classes = paired_ward_statistics(
        wards, zones, scenes[2016]["temperature"], scenes[2022]["temperature"], classes[2016], classes[2022],
        np.where(scenes[2016]["optical_valid"], scenes[2016]["features"][5], np.nan),
        np.where(scenes[2022]["optical_valid"], scenes[2022]["features"][5], np.nan))
    table.to_csv(out / "ward_statistics.csv", index=False)
    delta = np.where(paired, scenes[2022]["temperature"] - scenes[2016]["temperature"], np.nan)
    write_raster(raster_dir / "paired_lst_delta_c.tif", delta, profile)
    pixel_area = abs(profile["transform"].a * profile["transform"].e - profile["transform"].b * profile["transform"].d)
    transition_table(classes[2016], classes[2022], paired_classes, pixel_area).to_csv(out / "experimental_class_transitions.csv", index=False)
    old_grids = {}
    old_ranges = {}
    for year in (2016, 2022):
        for prefix in ("LULC_FINAL", "Clipped_LST"):
            path = root / "processed" / f"{prefix}_{year}.tif"
            if path.exists():
                used_inputs.append(path)
                old_grids[path.name] = grid_contract(path)
                if prefix == "Clipped_LST":
                    with rasterio.open(path) as src:
                        old_ranges[year] = describe(src.read(1, masked=True).compressed())
    print("Hashing exact input files", flush=True)
    manifest = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(set(used_inputs))]
    write_json(out / "input_manifest.json", {"dataset": "iith25ankit/urban-heat-island-uhi-analysis", "version": 1, "files": manifest})
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "Two selected Landsat daytime surface-temperature observations, not a seven-year time series or rural-reference UHI intensity",
        "ward_features": len(wards), "invalid_geometries_repaired": repairs,
        "reference_grid": {"crs": profile["crs"].to_string(), "width": profile["width"], "height": profile["height"],
                           "transform": list(profile["transform"])[:6], "pixel_area_m2": pixel_area,
                           "roi_pixels": int((zones > 0).sum()), "pixel_assignment": "pixel centre; last polygon wins at overlaps"},
        "config": {"seed": 42, "n_trees": 200, "block_size_m": block_size, "buffer_m": buffer_m, "n_spatial_folds": 5,
                   "qa_pixel_rejected_bits": [0, 1, 2, 3, 4, 5], "qa_radsat_rule": "reject nonzero", "resampling": "nearest",
                   "features": FEATURE_NAMES, "class_names_from_original_report": CLASS_NAMES},
        "scenes": {year: {"id": s["scene"], "metadata": s["metadata"], "quality": s["quality"],
                           "lst_c_all_valid": describe(s["temperature"]),
                           "lst_c_paired": describe(s["temperature"][paired]),
                           "st_uncertainty_k": describe(s["st_uncertainty_k"])} for year, s in scenes.items()},
        "paired_lst_delta_c": describe(delta), "training_provenance": provenance,
        "legacy_random_split_reproduction": legacy, "evaluation": evaluations,
        "exploratory_ward_associations": exploratory_correlations(table),
        "original_processed_grids": old_grids, "original_processed_lst_ranges": old_ranges,
        "land_cover_status": "Experimental diagnostics only: poor spatial validation and no retained 2016 water labels. Do not use transitions for reliable urban-conversion claims.",
        "limitations": ["Only two May daytime observations; weather and surface conditions can change between acquisitions",
                        "LST is surface temperature, not two-metre air temperature; no independent ground validation",
                        "Provided ward layer contains 145 features; completeness as a municipal inventory is not established",
                        "Manual training labels are spatially clustered, small and not independently verified",
                        "QA removes every 2016 water training point; conservative masking is retained rather than bypassed",
                        "Thermal data are delivered on a 30 m grid but Landsat 8 TIRS native thermal sampling is 100 m",
                        "ST uncertainty values do not cover all between-scene or modeling uncertainty",
                        "Ward associations are exploratory; no causal conclusion or naive IID significance claim"],
        "runtime": {"python": platform.python_version(), **{p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "rasterio", "geopandas", "pyogrio", "matplotlib", "joblib")}},
    }
    write_json(out / "results.json", result)
    if plots:
        plot_results(out, wards, profile, scenes, paired, evaluations)
    print(f"Complete: {out}", flush=True)
    return result
