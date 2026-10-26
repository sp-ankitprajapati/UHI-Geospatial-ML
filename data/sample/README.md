# Real labeled-point sample

These two tables contain the 157 and 161 QA-valid, labeled points retained from Kaggle dataset v1. Columns contain projected coordinates (EPSG:32644), source-row IDs, reported class labels, scaled B2/B3/B4/B5/B6 reflectance, NDVI and NDBI. The source point coordinates reproduce every original CSV SAMPLE column exactly. See `results/verified_run/results.json` for the verified band mapping.

The small tables support **model evaluation only** without downloading rasters. Run `uhi sample`. Full maps and ward temperature summaries require `uhi fetch` followed by `uhi run`.

Labels came from the original study, with no independent label audit. QA rejects all 41 water points in 2016, leaving no class 1 samples for that year's model. Do not treat these tables as representative accuracy samples for the full city. No municipal ward coverage or new ground truth was invented.
