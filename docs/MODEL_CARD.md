# Model card: experimental land-cover classifiers

**Task:** classify reported Water / Vegetation / Bare land / Urban low density / Urban high density labels. This is classification, not temperature regression.

**Intended use:** education, reproducibility, and diagnosing spatial sampling failure. **Status:** experimental only. Do not use modeled transitions to establish reliable urban growth areas, causal temperature drivers, planning decisions or population heat exposure.

**Data:** 203/168 original points; 157/161 remain after conservative QA and required-label cleanup. Labels reflect the original author's manual study with no independent audit. All 41 labeled water points in 2016 carry QA exclusions, leaving no retained water class. The model cannot learn that class for that year.

**Features:** fixed order blue, green, red, nir, swir1, ndvi, ndbi. No coordinates, year or thermal values enter the feature matrix. Coordinates define evaluation groups only. Five reflectance bands use the USGS scale and offset before index calculations.

**Algorithm:** separate annual RandomForestClassifier models, 200 trees, leaf minimum 2, balanced_subsample, random seed 42. No test-guided tuning. Saved joblib files must only be loaded from trusted runs; models include sidecar feature/class/runtime metadata in the run outputs.

| Year | Random accuracy | Random macro F1 | Spatial OOF accuracy | Spatial OOF macro F1 | Spatial groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2016 | 1.0000 | 0.8000 | 0.2675 | 0.1313 | 9 |
| 2022 | 1.0000 | 1.0000 | 0.1801 | 0.2254 | 12 |

Five-fold grouped tests use 1 km blocks and a 300 m exclusion around test points. Predictions, role assignments, missing training classes, confusion matrices and per-class metrics are published in `results/verified_run`. A perfect random split on clustered points is not evidence of city-wide accuracy. Scores from different split designs measure different generalization questions and are not a claimed performance improvement.

**Next valid model improvement:** collect independently checked examples for every class across many separated locations and both dates. Preserve geographic holdout regions before feature/hyperparameter work. Reassess block and buffer scales from spatial dependence and intended deployment. Audit seasonal/label definitions and cross-date comparability. Treat adding deep learning or more trees as a hypothesis to test, not an automatic remedy for missing information.
