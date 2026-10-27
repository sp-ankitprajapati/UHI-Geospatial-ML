import numpy as np
import pandas as pd
import pytest

from uhi.features import (FEATURE_NAMES, feature_rows, normalized_difference,
                          optical_features, quality_mask, restore_labels,
                          surface_temperature, validate_training)
from uhi.analysis import paired_ward_statistics, predict_masked, transition_table
from uhi.evaluation import buffered_training_indices, evaluate, spatial_groups


def test_nullable_id_does_not_delete_good_training_rows():
    frame = pd.DataFrame({"id": [np.nan, np.nan, np.nan], "class": [1, 2, np.nan], "blue": [3, 4, 5]})
    assert len(validate_training(frame, ["blue"])) == 2


@pytest.mark.parametrize("bad", [0, 6, 2.5, np.inf])
def test_invalid_class_ids_fail(bad):
    with pytest.raises(ValueError):
        validate_training(pd.DataFrame({"class": [bad], "blue": [1]}), ["blue"])


@pytest.mark.parametrize("bit", range(6))
def test_bad_quality_flags_are_rejected(bit):
    assert not quality_mask(np.array([1 << bit], dtype="uint16"))[0]


def test_water_clear_flag_is_retained_and_saturation_rejected():
    qa = np.array([128 + 64, 64], dtype="uint16")
    assert quality_mask(qa).tolist() == [True, True]
    assert quality_mask(qa, np.array([0, 64])).tolist() == [True, False]


def test_temperature_scaling_and_fill_are_separate():
    result = surface_temperature(np.array([0, 45000, 45000]), np.array([True, True, False]))
    assert np.isnan(result[[0, 2]]).all()
    assert result[1] == pytest.approx(29.6609, abs=.0001)


def test_indices_apply_reflectance_offset_before_division():
    raw = np.array([[10000], [11000], [12000], [18000], [16000]])
    f = optical_features(raw)
    nir, red = 18000 * .0000275 - .2, 12000 * .0000275 - .2
    assert f.shape == (len(FEATURE_NAMES), 1)
    assert f[5, 0] == pytest.approx((nir - red) / (nir + red))
    assert f[5, 0] != pytest.approx((18000 - 12000) / (18000 + 12000))
    assert np.isnan(normalized_difference([1], [-1]))[0]


def test_masked_prediction_preserves_pixel_positions():
    class Threshold:
        def predict(self, rows):
            return np.where(rows[:, 0] < 3, 1, 2)
    cube = np.array([[[1., 99., 3.], [2., 4., 99.]]])
    mask = np.array([[True, False, True], [True, True, False]])
    expected = np.array([[1, 0, 2], [1, 2, 0]])
    np.testing.assert_array_equal(predict_masked(Threshold(), cube, mask, batch_size=1), expected)
    np.testing.assert_array_equal(predict_masked(Threshold(), cube, mask, batch_size=100), expected)
    with pytest.raises(ValueError):
        restore_labels([1], mask)
    with pytest.raises(ValueError):
        feature_rows(cube, np.zeros((1, 2), bool))


def test_buffer_excludes_neighbor_across_block_boundary():
    xy = np.array([[999., 0.], [1001., 0.], [1800., 0.]])
    groups = spatial_groups(xy, 1000)
    assert groups[0] != groups[1]
    train = buffered_training_indices(np.array([1, 2]), np.array([0]), xy, 300)
    assert train.tolist() == [2]


def test_every_spatial_sample_is_held_out_once_without_group_leakage():
    xy = np.c_[np.arange(25) * 1200, np.zeros(25)]
    X, y = np.c_[np.arange(25), np.arange(25) % 2], np.tile(np.arange(1, 6), 5)
    _, predictions, assignments = evaluate(X, y, xy, n_splits=5)
    spatial = predictions.query("split == 'spatial_buffered' and model == 'random_forest'")
    assert sorted(spatial.sample_index.tolist()) == list(range(25))
    for _, fold in assignments.query("split == 'spatial_buffered'").groupby("fold"):
        assert not set(fold.query("role == 'train'").group) & set(fold.query("role == 'test'").group)


def test_ward_changes_use_identical_pixel_support():
    wards = pd.DataFrame({"ward_id": ["a", "b"], "ward_name": ["A", "B"]})
    zones = np.array([[1, 1], [2, 2]])
    old = np.array([[10., 100.], [20., 30.]])
    new = np.array([[12., np.nan], [25., 40.]])
    classes = np.array([[3, 0], [4, 5]], dtype="uint8")
    table, paired, _ = paired_ward_statistics(wards, zones, old, new, classes, classes,
                                              np.ones((2, 2)), np.ones((2, 2)) * 2)
    assert table.loc[0, "lst_2016_mean_c"] == 10
    assert table.loc[0, "lst_delta_mean_c"] == 2
    assert table.loc[0, "paired_lst_fraction"] == .5
    assert table.loc[1, "lst_delta_mean_c"] == 7.5
    assert paired.sum() == 3


def test_nodata_does_not_become_a_land_conversion():
    first, second = np.array([[3, 0]]), np.array([[4, 5]])
    table = transition_table(first, second, first > 0, 900)
    assert table.pixels.sum() == 1
    assert table.map_grid_area_km2.sum() == pytest.approx(.0009)
    with pytest.raises(ValueError):
        transition_table(first, second, np.ones((1, 2), bool), 900)
