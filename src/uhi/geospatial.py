"""Read known products onto one reference grid, preserving masks and geometry."""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window, rasterize
from rasterio.vrt import WarpedVRT

from .features import optical_features, quality_mask, surface_temperature

SCENE_IDS = {
    2016: "LC08_L2SP_144048_20160523_20200906_02_T1",
    2022: "LC08_L2SP_144048_20220524_20220602_02_T1",
}


def scene_paths(root, year):
    scene = SCENE_IDS[year]
    folder = Path(root) / "data_raw_bands" / scene
    names = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ST_B10",
             "QA_PIXEL", "QA_RADSAT", "ST_QA"]
    paths = {name: folder / f"{scene}_{name}.TIF" for name in names}
    paths["metadata"] = folder / f"{scene}_MTL.json"
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Required scene inputs missing: {missing}")
    return paths


def load_wards(path, crs):
    wards = gpd.read_file(path)
    if wards.crs is None or wards.empty or wards.geometry.isna().any():
        raise ValueError("Nonempty ward geometries and a source CRS are required")
    wards = wards.to_crs(crs).copy()
    if not wards.crs.is_projected or wards.crs.axis_info[0].unit_name not in {"metre", "meter"}:
        raise ValueError("Analysis requires projected metre coordinates")
    repaired = int((~wards.geometry.is_valid).sum())
    wards.geometry = wards.geometry.make_valid()
    if wards.geometry.is_empty.any() or not wards.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Ward repair must produce nonempty polygon geometries")
    id_field = "@id" if "@id" in wards else "True_Ward" if "True_Ward" in wards else None
    if id_field is None or wards[id_field].isna().any() or wards[id_field].duplicated().any():
        raise ValueError("A stable, nonmissing, unique ward ID is required")
    wards["ward_id"] = wards[id_field].astype(str)
    wards["ward_name"] = wards["name"].astype(str) if "name" in wards else wards.ward_id
    return wards[["ward_id", "ward_name", "geometry"]], repaired


def reference_grid(path, wards):
    with rasterio.open(path) as src:
        if src.crs != wards.crs:
            raise ValueError("Reference raster and ward CRS must agree")
        window = geometry_window(src, list(wards.geometry))
        profile = {"driver": "GTiff", "height": int(window.height), "width": int(window.width),
                   "crs": src.crs, "transform": src.window_transform(window)}
    zones = rasterize([(g, i + 1) for i, g in enumerate(wards.geometry)],
                      out_shape=(profile["height"], profile["width"]),
                      transform=profile["transform"], fill=0, dtype="int32")
    if not (zones > 0).any():
        raise ValueError("No pixel centres fall in the supplied wards")
    return profile, zones


def read_on_grid(path, profile):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Missing source CRS: {path}")
        # QA_RADSAT=0 means "no saturation". It must never become destination
        # NoData: the GDAL warper may replace valid zeros with one in that case.
        destination_nodata = src.nodata if src.nodata is not None else np.iinfo(src.dtypes[0]).max
        with WarpedVRT(src, crs=profile["crs"], transform=profile["transform"],
                       height=profile["height"], width=profile["width"],
                       resampling=Resampling.nearest, nodata=destination_nodata) as vrt:
            return vrt.read(1)


def read_scene(root, year, profile, roi):
    paths = scene_paths(root, year)
    metadata = json.loads(paths["metadata"].read_text())["LANDSAT_METADATA_FILE"]
    st_params = metadata["LEVEL2_SURFACE_TEMPERATURE_PARAMETERS"]
    scale = float(st_params["TEMPERATURE_MULT_BAND_ST_B10"])
    offset = float(st_params["TEMPERATURE_ADD_BAND_ST_B10"])
    # Fail on unsupported products rather than applying an unexplained constant.
    sr_params = metadata["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]
    for band in (2, 3, 4, 5, 6):
        if not np.isclose(float(sr_params[f"REFLECTANCE_MULT_BAND_{band}"]), 0.0000275):
            raise ValueError("Unsupported reflectance scale")
        if not np.isclose(float(sr_params[f"REFLECTANCE_ADD_BAND_{band}"]), -0.2):
            raise ValueError("Unsupported reflectance offset")
    raw = np.stack([read_on_grid(paths[f"SR_B{band}"], profile) for band in (2, 3, 4, 5, 6)])
    qa = read_on_grid(paths["QA_PIXEL"], profile)
    rad = read_on_grid(paths["QA_RADSAT"], profile)
    clear = roi & quality_mask(qa, rad)
    features = optical_features(raw)
    optical_valid = clear & (raw > 0).all(axis=0) & np.isfinite(features).all(axis=0)
    st_dn = read_on_grid(paths["ST_B10"], profile)
    temperature = surface_temperature(st_dn, clear & (st_dn >= 293), scale, offset)
    if not optical_valid.any() or not np.isfinite(temperature).any():
        raise ValueError(f"No valid study pixels remain for {year}; inspect grids and QA before reporting results")
    st_uncertainty = read_on_grid(paths["ST_QA"], profile).astype(float) * 0.01
    st_uncertainty[~np.isfinite(temperature)] = np.nan
    return {"features": features, "optical_valid": optical_valid, "temperature": temperature,
            "st_uncertainty_k": st_uncertainty,
            "scene": SCENE_IDS[year], "metadata": metadata["IMAGE_ATTRIBUTES"],
            "quality": {"roi_pixels": int(roi.sum()), "clear_pixels": int(clear.sum()),
                        "optical_valid_pixels": int(optical_valid.sum()),
                        "lst_valid_pixels": int(np.isfinite(temperature).sum()),
                        "st_scale": scale, "st_offset_k": offset}}


def grid_contract(path):
    with rasterio.open(path) as src:
        return {"crs": src.crs.to_string() if src.crs else None, "width": src.width,
                "height": src.height, "transform": list(src.transform)[:6],
                "nodata": src.nodata, "count": src.count, "dtypes": list(src.dtypes)}


def write_raster(path, array, profile, labels=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = dict(profile, count=1, dtype="uint8" if labels else "float32",
                nodata=0 if labels else np.nan, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **info) as dst:
        dst.write(np.asarray(array, dtype=info["dtype"]), 1)
        dst.update_tags(analysis="Two selected observations, not an annual climate trend",
                        units="class_id" if labels else "see filename and result manifest")
