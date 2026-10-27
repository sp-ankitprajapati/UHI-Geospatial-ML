import numpy as np
import rasterio
from rasterio.transform import from_origin
from uhi.geospatial import read_on_grid, write_raster


def test_shifted_grid_matches_coordinates_and_preserves_valid_qa_zero(tmp_path):
    path = tmp_path / "qa.tif"
    with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1,
                       dtype="uint16", crs="EPSG:32644", transform=from_origin(30, 60, 30, 30)) as dst:
        dst.write(np.array([[0, 64], [0, 0]], dtype="uint16"), 1)
    profile = {"crs": "EPSG:32644", "transform": from_origin(0, 60, 30, 30), "height": 2, "width": 3}
    actual = read_on_grid(path, profile)
    np.testing.assert_array_equal(actual, [[65535, 0, 64], [65535, 0, 0]])


def test_output_nan_and_class_nodata_round_trip(tmp_path):
    profile = {"crs": "EPSG:32644", "transform": from_origin(0, 60, 30, 30), "height": 2, "width": 2}
    path = tmp_path / "lst_c.tif"
    write_raster(path, [[20, np.nan], [30, 40]], profile)
    with rasterio.open(path) as src:
        assert np.isnan(src.nodata)
        assert src.read(1, masked=True).count() == 3
    path = tmp_path / "classes.tif"
    write_raster(path, [[1, 0], [2, 5]], profile, labels=True)
    with rasterio.open(path) as src:
        assert src.nodata == 0
        assert src.read(1, masked=True).count() == 3
