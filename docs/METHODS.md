# Methods and interpretation

## Data and observation design

Kaggle v1 contains the original study materials. This run uses Landsat 8 Collection 2 Level-2 scenes `LC08_L2SP_144048_20160523_20200906_02_T1` and `LC08_L2SP_144048_20220524_20220602_02_T1`. Scene metadata reports 05:09:17 and 05:09:36 UTC, respectively. Acquisition dates provide two observations separated by six years. They do not form seven annual observations.

The provided ward GeoJSON contains 145 polygon features. Geometry repair reports zero invalid inputs. The analysis uses their supplied extent without reconstructing missing municipal wards. Output ward data retains stable IDs and names, dropping unrelated OSM editor metadata.

## Common grid and quality

The 2016 reference establishes EPSG:32644 and 30 m cells. Cropping to the ward envelope gives 1,235 columns by 995 rows, with origin (210165, 1943475). Pixel-centre polygon assignment includes 610,996 grid cells. Rasterization assigns overlapping pixel centres to the last polygon in source order. Area figures are map-grid areas, not an independent survey of municipal area.

All 2022 bands and QA arrays use nearest-neighbor coordinate alignment on that exact grid. Original full-scene classification origins differ by 300 m in easting. Equal raster shapes alone cannot authorize pixel subtraction. Nearest resampling preserves stored DN and categorical bits; the aligned 30 m grids have no subpixel shift here.

QA_PIXEL bits 0 through 5 exclude fill, dilated cloud, cirrus, cloud, cloud shadow and snow. Water bit 7 alone remains valid. A conservative QA_RADSAT rule rejects any nonzero condition. NoData zero in reflectance/ST remains invalid. **Zero is a valid QA_RADSAT value**, so its warp destination uses a separate sentinel; a regression test protects this distinction. ST requires stored DN >= 293 and a valid QA mask. No arbitrary ST_QA threshold was selected after viewing results. Product ST uncertainty is exported at 0.01 K per stored unit.

## Physical quantities and feature contract

MTL metadata supplies ST scale and offset. Celsius = DN × 0.00341802 + 149 - 273.15. These inputs are already Level-2 surface temperature, so a Level-1 radiance/brightness-temperature conversion would be the wrong workflow.

Both classifiers use B2/B3/B4/B5/B6 surface reflectance, NDVI = (NIR - red)/(NIR + red), and NDBI = (SWIR1 - NIR)/(SWIR1 + NIR). Reflectance = DN × 0.0000275 - 0.2 **before** computing normalized differences. The additive offset does not cancel from the denominator. Zero denominators become missing. Thermal ST does not enter the new land-cover model, reducing circularity in exploratory temperature comparisons.

Source point coordinates reproduce every original CSV feature exactly. The 2016 CSV contains B2,B3,B4,B5,B6,ST_B10; the 2022 CSV contains B4,B3,B2. The shared modern feature contract therefore constitutes a new experiment, distinct from the legacy model reproduction.

## Models and evaluation

The legacy baseline uses the original 100-tree RF and unstratified 80/20 seed-42 split after dropping missing required features/labels only. Both old CSVs have null ID columns, making blanket `dropna()` unsafe. The baseline reproduces 100% accuracy on 40/34 row-holdout points.

The new experiment uses 200 trees, minimum leaf size 2, balanced bootstrap class weights and seed 42. QA/label validation retains 157/161 samples. No hyperparameter search used the holdout folds. The modern random 80/20 split is stratified. Spatial evaluation uses floor(x/1000), floor(y/1000) groups, GroupKFold with five folds, and excludes training samples within 300 m of any test point. These distances are an explicit demonstration design, not a statistically estimated decorrelation range. The limited labels occupy only 9/12 groups. Some folds lack entire classes after holdout and buffering. The majority-class baseline also performs poorly when a class occupies a held-out cluster.

Each retained sample appears once in the spatial test predictions. Accuracy and fixed-five-class macro F1 aggregate all out-of-fold predictions. Fold mean and standard deviation are separate descriptive values. Missing 2016 water labels produce a zero F1 term even in its perfect random row split, giving macro F1 0.8. This explicitly exposes incomplete class coverage.

Final maps fit all retained points after evaluation. Their status is experimental because the spatial tests and class coverage fail to support reliable map generalization. Impurity feature importance is diagnostic only and cannot establish temperature drivers.

## Paired summaries and association

Each ward's temperature values use identical locations valid on both dates. Mean, median, p10 and p90 complement coverage fractions. Delta uses pixelwise 2022 minus 2016 values. City/study-area mean uses pixels directly and is not an unweighted average of ward means. NDVI comparisons require their own paired optical support. Class transitions use nonzero labels on both dates, report their separate support, and remain experimental.

Exploratory ward Pearson r values use at least 80% paired LST coverage (139 wards), with explicit variable names. Predictor supports can differ slightly from LST support. No naive IID p-values or causal ranking are reported because adjacent wards and modeled errors are spatially dependent. No independent weather normalization, rural reference or ground validation was performed. Between-date differences cannot establish a long-term heat trend or a conversion-driven causal effect.

## References

- [USGS Level-2 scaling](https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products)
- [USGS Collection 2 QA bits](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands)
- [NASA Landsat 8 instrument resolution](https://science.nasa.gov/mission/landsat-8/)
- [Rasterio coordinate reprojection](https://rasterio.readthedocs.io/en/stable/topics/reproject.html)
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [SciPy Pearson correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)

Figure colour limits use paired 2nd/98th percentiles for display; saved temperature values remain unclipped. The difference map uses symmetric limits.
