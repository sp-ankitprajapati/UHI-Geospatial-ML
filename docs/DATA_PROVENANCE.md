# Data provenance

Source: [Kaggle urban-heat-island-uhi-analysis, version 1](https://www.kaggle.com/datasets/iith25ankit/urban-heat-island-uhi-analysis/data). Baseline code: [commit 4bdf49cf](https://github.com/sp-ankitprajapati/UHI-Geospatial-ML/tree/4bdf49cf58589048d06a2e4c97fa34a17d2a6905).

`src/uhi/input_files.json` lists each requested Kaggle file name, local relative path, byte length and SHA-256. The public single-file endpoint pins datasetVersionNumber=1. Hashes were computed from the exact inputs of the checked run and checked against subsequent local fetch reuse. A changed download fails rather than silently becoming a new dataset version.

Raw optical B2-B6, ST_B10, QA_PIXEL, QA_RADSAT, ST_QA and MTL JSON supply physical values. Labeled shapefiles supply coordinates and reported classes. The ward GeoJSON supplies polygon IDs and geometry. CSVs verify original band mapping and reproduce the legacy split. Original LULC_FINAL and clipped-LST products are audit comparators only, not the source of newly calculated LST or classification.

The original report supplies class meanings. No independent label validation, seven-date imagery collection, complete 150-ward inventory, rural reference or weather ground-truth series was found/introduced in this upgrade. Original data ownership and dataset reuse terms remain with their respective sources; no new license grant for imagery or boundaries is asserted.

Bundled sample tables contain scientific feature values, study class IDs and spatial sample coordinates. Output ward data excludes unrelated OSM contributor attributes. `results/verified_run/input_manifest.json` ties published evidence to exact inputs. `results.json` records the execution environment, configuration and observation dates. Generated artifacts are deterministic for fixed data/configuration apart from run timestamps and possible library/platform numerical differences.
