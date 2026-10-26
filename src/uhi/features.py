"""Explicit Landsat Collection 2 contracts and array transforms."""

import numpy as np

FEATURE_NAMES = ("blue", "green", "red", "nir", "swir1", "ndvi", "ndbi")
CLASS_NAMES = {1: "Water", 2: "Vegetation", 3: "Bare land",
               4: "Urban low density", 5: "Urban high density"}
# Class meanings are recovered from the original report, not independent truth labels.
REFLECTANCE_SCALE = 0.0000275
REFLECTANCE_OFFSET = -0.2
ST_SCALE = 0.00341802
ST_OFFSET_K = 149.0


def quality_mask(qa_pixel, qa_radsat=None):
    """Reject L8/9 C2 fill/cloud/cirrus/shadow/snow; retain water.

    Conservatively reject any nonzero QA_RADSAT condition when supplied.
    This does not claim to remove every atmospheric or emissivity error.
    """
    qa = np.asarray(qa_pixel)
    if not np.issubdtype(qa.dtype, np.integer):
        raise ValueError("QA_PIXEL must contain integers")
    if np.any((qa < 0) | (qa > 65535)):
        raise ValueError("QA_PIXEL must fit uint16")
    valid = (qa & 0b111111) == 0
    if qa_radsat is not None:
        rad = np.asarray(qa_radsat)
        if rad.shape != qa.shape:
            raise ValueError("QA arrays must share a grid")
        valid &= rad == 0
    return valid


def normalized_difference(first, second):
    first, second = np.broadcast_arrays(np.asarray(first, dtype=float),
                                         np.asarray(second, dtype=float))
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-12)
    result = np.full(first.shape, np.nan, dtype=np.float32)
    np.divide(first - second, denominator, out=result, where=valid)
    return result


def surface_temperature(dn, valid=None, scale=ST_SCALE, offset=ST_OFFSET_K):
    """Stored C2 Level-2 ST DN -> Celsius, not a Level-1 radiance algorithm."""
    dn = np.asarray(dn)
    mask = np.isfinite(dn) & (dn > 0)
    if valid is not None:
        if np.shape(valid) != dn.shape:
            raise ValueError("ST validity mask must match the array")
        mask &= valid
    return np.where(mask, dn * scale + offset - 273.15, np.nan).astype(np.float32)


def optical_features(raw_bands):
    """B2,B3,B4,B5,B6 stored DN -> five reflectances, NDVI, NDBI."""
    raw = np.asarray(raw_bands)
    if raw.ndim < 2 or raw.shape[0] != 5:
        raise ValueError("Expected five bands in B2,B3,B4,B5,B6 order")
    sr = raw.astype(np.float32) * REFLECTANCE_SCALE + REFLECTANCE_OFFSET
    ndvi = normalized_difference(sr[3], sr[2])
    ndbi = normalized_difference(sr[4], sr[3])
    return np.concatenate([sr, ndvi[None], ndbi[None]], axis=0)


def validate_training(frame, feature_names, label="class"):
    """Drop rows only for required label/features, never for nullable IDs."""
    names = list(feature_names)
    if not names or len(set(names)) != len(names) or label in names:
        raise ValueError("Feature contract must be unique and exclude the target")
    missing = [c for c in [label] + names if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    clean = frame.dropna(subset=[label] + names).copy()
    if clean.empty:
        raise ValueError("No samples remain after required-column validation")
    values = clean[names].to_numpy(dtype=float)
    labels = clean[label].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError("Training values must be finite")
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Class IDs must be integers")
    if not set(labels.astype(int)).issubset(CLASS_NAMES):
        raise ValueError("Unknown land-cover class ID")
    return clean


def feature_rows(cube, valid):
    cube, valid = np.asarray(cube), np.asarray(valid, dtype=bool)
    if cube.ndim != 3 or cube.shape[1:] != valid.shape:
        raise ValueError("Expected (features, rows, cols) and matching validity mask")
    if not np.isfinite(cube[:, valid]).all():
        raise ValueError("A valid pixel contains nonfinite features")
    return cube[:, valid].T


def restore_labels(labels, valid):
    valid = np.asarray(valid, dtype=bool)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != int(valid.sum()):
        raise ValueError("Label count must match valid pixels")
    if not set(np.unique(labels)).issubset(CLASS_NAMES):
        raise ValueError("Output class IDs must be in 1..5; zero is reserved NoData")
    result = np.zeros(valid.shape, dtype=np.uint8)
    result[valid] = labels
    return result
