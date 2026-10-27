# UHI project: zero se actual code aur interview tak

## Is guide ko kaise use karna hai

Pehle project ka flow samjho, phir har function ka input/output bolo, phir apne haath se tiny example likho. Padh lena aur khud code explain kar pana alag cheezein hain. Tumhe poora project scratch se dobara banana zaroori nahi. Neeche ke small parts khud implement karna zaroori practice hai.

Yeh guide **upgraded code aur actual rerun** explain karti hai. Pehle wale pack ka original audit historical snapshot hai. Latest numbers `results/verified_run/results.json` mein hain. Humne engineering aur evaluation improve ki hai. Poor spatial score ko high accuracy bana dene ka claim nahi kiya.

## 1. Project karta kya hai?

Satellite se Hyderabad ke surface ka data aata hai. Hum do dates ke temperature maps ko same locations par compare karte hain. Phir ward-wise summary nikalte hain. Ek separate Random Forest model har pixel ko land-cover class dene ki koshish karta hai.

LST = land surface temperature. Road ki surface dhoop mein 50 °C ho sakti hai, jabki weather app hawa ka alag temperature bataye. Humne weather-station air temperature validate nahi kiya. Aur true urban-rural UHI intensity ke liye defined rural reference chahiye; current pipeline woh metric calculate nahi karti. Repository ka naam historical UHI study hai, exact implemented scope two-scene surface-temperature comparison hai.

Dates: 23 May 2016 aur 24 May 2022. Do observations hain. Har saal ka data nahi hai. Boundary file mein 145 features hain. 150 bolna supplied file ka actual count explain nahi karega.

**Interview line:** “Maine do Landsat observations ke liye reproducible geospatial pipeline banayi, common clear pixels par ward temperature summaries nikali aur spatial validation se classifier ki sampling limitations identify ki.” Is line ko tab use karo jab tum code walkthrough aur exercises kar chuke ho.

## 2. Raster, vector, band aur pixel

Raster numbers ka grid hai. Ek band ko 2D array samjho: rows aur columns. Multiple bands ko stack karo toh shape `(bands, height, width)` banegi. Vector boundary points/lines/polygons se area describe karti hai. Ward polygon batata hai ki map ke kis area ki summary chahiye.

Landsat optical B2 blue, B3 green, B4 red, B5 near-infrared, B6 shortwave infrared hai. Camera ke RGB se extra wavelength information milti hai. Plants NIR aur red mein different response dete hain, isliye NDVI useful hota hai.

Stored DN ek encoded number hai. Bina scale/offset lagaye usko reflectance ya degrees samajhna galat hoga. Delivered grid 30 m hai, har map-grid cell 900 m². Thermal sensor ka native sampling 100 m hai. 30 m grid pe resampling se nayi 30 m thermal detail automatically create nahi hoti.

**Practice:** `(7, 100, 200)` shape mein kitne feature vectors hain aur har vector ki length kya hai? Answer: 20,000 vectors, length 7, agar sab pixels valid hon.

## 3. CRS, transform aur alignment

CRS coordinate system batata hai. Latitude/longitude degrees mein hote hain. EPSG:32644 projected UTM coordinates metres mein deta hai, isliye 300 m buffer meaningful hai. Affine transform grid column/row ko ground x/y se connect karta hai.

North-up 30 m grid ke pixel-centre ka formula:

```python
x = left + (column + 0.5) * 30
y = top - (row + 0.5) * 30
```

Do arrays ki shape same ho sakti hai, lekin unka starting x alag ho sakta hai. Original full-scene maps ka x origin 300 m alag tha. Agar index se direct subtract karoge toh different locations compare ho sakti hain. `reference_grid` 2016 se common grid banata hai; `read_on_grid` 2022 ko usi CRS, transform, width, height par align karta hai.

Nearest neighbor closest source pixel choose karta hai. QA bits aur class labels ke liye averaging meaningless ho sakti hai. Continuous data par bilinear kabhi useful hota hai, lekin is run mein already aligned 30 m grid ke DN aur QA ko nearest se preserve kiya.

**Khud bolo:** shape, CRS, transform aur resolution mein kya difference hai? In chaaron ko verify kiye bina subtraction kyun risky hai?

## 4. QA mask: kharab observation ko pehchanna

QA = quality assessment. QA_PIXEL ek integer hai jiske bits alag flags store karte hain. Binary 8 ka bit 3 on hai. Code `qa & (1 << 3)` check karega ki cloud flag set hai ya nahi. Hum bits 0..5 reject karte hain. Water bit 7 apne aap bad observation nahi hai.

```python
clear = (qa_pixel & 0b111111) == 0
valid = clear & (qa_radsat == 0)
```

`&` NumPy arrays par elementwise AND hai. Normal Python `and` poori array ka ek truth value maangta hai aur ambiguous ho sakta hai. `valid` ki shape `(height, width)` hai. True pixels analysis mein jayenge; False pixels NoData rahenge.

Important real bug: QA_RADSAT mein zero ka matlab valid/no saturation hai. Agar warp mein zero ko destination NoData bana diya, valid zero badal sakta hai. Updated reader separate sentinel use karta hai, aur shifted-grid test exact expected values check karta hai. NoData ka meaning data type aur band se depend karta hai.

2016 ke 41 labeled water points QA exclusions carry karte hain. Mask hata kar score bachana defensible fix nahi. Ab 2016 classifier ke paas water example nahi bacha. Is limitation ko model card mein likha hai. Water actual cloudy tha ya QA misclassification thi, woh independent evidence ke bina decide nahi kiya.

## 5. Temperature calculation aur deterministic code

Level-2 ST_B10 already surface-temperature product hai. Metadata se scale aur offset milte hain:

```python
celsius = stored_dn * 0.00341802 + 149.0 - 273.15
```

DN 45,000 ke liye 29.6609 °C aayega. Zero fill ko formula mein daal kar -124.15 °C report karna wrong hoga. Pehle mask, phir calculation. Updated code invalid cells ko `np.nan` banata hai. Celsius minus Celsius ka difference degree Celsius mein hi meaningful hai.

`surface_temperature` deterministic function hai. Same valid input aur constants par same answer aata hai. RF ke predicted label ka LST formula mein koi role nahi. Isi separation se classifier poor ho tab bhi direct temperature workflow ko independently audit kar sakte hain.

ST_QA 0.01 K units mein product uncertainty deta hai. Yeh between-year weather differences ya har model error cover nahi karta. Hum uncertainty raster save karte hain; fabricated confidence interval nahi banate.

## 6. Feature engineering: NDVI aur NDBI

Feature model ko diya gaya input column hai. Target woh class hai jo predict karni hai. Dono years ke fixed features: blue, green, red, nir, swir1, ndvi, ndbi. Feature order training aur inference dono mein same hona chahiye.

```python
reflectance = dn * 0.0000275 - 0.2
ndvi = (nir - red) / (nir + red)
ndbi = (swir1 - nir) / (swir1 + nir)
```

Formula mein nir/red **scaled reflectance** hain. Additive offset denominator se cancel nahi hota. Isliye raw DN par ratio aur scaled ratio alag ho sakte hain. Zero denominator par `nan` banao, infinity ko model mein mat bhejo. Negative reflectance/noisy pixels ki interpretation bhi cautious honi chahiye; index ek magic ground-truth label nahi.

Original 2016 CSV mein six bands, including thermal ST tha. 2022 CSV mein sirf red/green/blue the. Humne raw rasters ko source labeled coordinates par sample karke har old CSV column ki exact band identity verify ki. New model dono dates par same seven optical features use karta hai. Thermal feature hatane se temperature associations mein obvious circularity reduce hoti hai.

## 7. Pandas cleaning aur schema

DataFrame table hai. `frame[features]` selected columns deta hai. `to_numpy()` ML-compatible numeric array deta hai. `dropna()` har column check karta hai. Agar optional `id` column completely blank ho, saari rows gayab ho sakti hain.

```python
required = ['class'] + feature_names
clean = frame.dropna(subset=required)
```

Updated validator missing columns, infinite values, noninteger labels aur unknown classes reject karta hai. Class IDs 1..5 valid hain; zero output raster mein NoData ke liye reserved hai. “Code run ho gaya” data contract correct hone ka proof nahi. Misordered bands finite numbers honge, phir bhi prediction wrong ho sakti hai.

**Practice:** do valid rows, blank ID aur ek missing class wali table banao. Required-column cleanup ke baad do rows bachni chahiye. Is project ka test yehi real failure protect karta hai.

## 8. Random Forest samajhna

Decision tree feature thresholds se branching questions poochta hai. Random Forest many trees train karta hai aur votes combine karta hai. Different bootstrap samples aur feature subsets tree errors ko less correlated banane ki koshish karte hain. Trees ko usually feature scaling ki need linear-distance models jaisi nahi hoti; yahan physical reflectance conversion index correctness aur shared feature meaning ke liye hai.

`n_estimators=200`: trees ki count. `min_samples_leaf=2`: leaf mein kam se kam do training samples. `class_weight='balanced_subsample'`: each bootstrap sample ke class frequencies se weights. `random_state=42`: repeatable randomness. `n_jobs=-1`: available CPU cores use karta hai. Zyada trees missing labels ko create nahi karte.

Classifier class 1/2/3/4/5 predict karta hai. Temperature regression continuous degrees predict karega, jiske metrics MAE/RMSE ho sakte hain. Current project ko temperature prediction ML model bolna incorrect hai.

## 9. Random split aur spatial validation

Training se model patterns seekhta hai; test se unseen examples par estimate milta hai. Random split rows mix karta hai. Agar same lake/patch ke neighboring pixels train aur test dono mein aa gaye toh test easy ho sakta hai. Isko spatial dependence se optimistic evaluation bolte hain. Sirf feature matrix se coordinates drop karna is dependency ko remove nahi karta.

`spatial_groups` 1 km cells assign karta hai: floor(x/1000), floor(y/1000). `GroupKFold` test cell ke samples ko same fold mein rakhta hai. `buffered_training_indices` test points se 300 m ke andar training points ko remove karta hai, even agar adjacent block mein hon. KD-tree nearest distance efficiently query karta hai.

2016 labels sirf 9 groups aur 2022 labels 12 groups cover karte hain. Five-fold test mein kabhi poori class training se absent ho jaati hai. Isliye failure sirf hyperparameters ka issue nahi, sampling design ka evidence hai. 1 km/300 m current explicit design choices hain, estimated decorrelation distances nahi. Intended deployment aur spatial dependence se future distances justify karne honge.

**Real results:** modern random accuracy 100%/100%, spatial OOF accuracy 26.75%/18.01%. Isko “accuracy improved” nahi bolna. “Evaluation ko realistic banaya aur model ki limitation expose ki” bolna correct hai.

## 10. Accuracy, precision, recall, F1 aur baseline

Accuracy = correct predictions / test examples. Precision = predicted class mein kitne correct. Recall = actual class ke kitne detect hue. F1 precision aur recall ka harmonic mean: `2*P*R/(P+R)`. Macro F1 har class ko equal weight deta hai; weighted F1 frequent classes ko zyada weight deta hai.

Confusion matrix rows actual aur columns predicted classes hain. Off-diagonal values misclassifications dikhati hain. Majority baseline hamesha training ki most common class bolta hai. RF ko is simple baseline ke against compare karna useful hai. Spatial baseline ka poor score bhi held-out cluster mein different class dominance dikha sakta hai.

Hum fixed five-class macro F1 use karte hain. 2016 random test ke retained four classes perfect hain, lekin missing water class F1 zero hai. Isliye accuracy 1.0 aur macro F1 0.8 ek saath aa sakte hain. Exact label set aur absent-class policy bolna zaroori hai.

Out-of-fold = har sample us fold ke model se predict hota hai jiski training mein woh sample nahi tha. OOF accuracy sab held-out predictions aggregate karke nikalti hai. Unequal fold sizes mein fold accuracies ka simple average same quantity nahi hota.

## 11. NumPy reshaping aur masked inference

Cube `(features, height, width)` se valid rows `cube[:, valid].T` banti hain. Shape `(valid_pixel_count, feature_count)` hoti hai, sklearn ko yehi chahiye. `.T` axes transpose karta hai. Boolean mask invalid features ko inference se bahar rakhta hai.

`predict_masked` 65,536 rows ke batches mein predictions karta hai aur same valid positions par restore karta hai. Output zero se initialize hota hai; False pixels zero NoData rahte hain. Test batch size 1 aur large batch ka identical output verify karta hai.

Memory honestly explain karo: source rasters cropped AOI par padhe jaate hain. Feature arrays aur valid row matrix abhi memory mein materialize hote hain; sirf prediction batches bounded hain. Full source-scene streaming ka claim nahi. Bahut bade AOI ke liye window-by-window read/feature/predict/write extension chahiye.

## 12. Ward statistics aur paired support

Ward ko region mask samjho. Same ward ke 2016 aur 2022 means agar different valid locations se nikle toh selection difference answer badal sakta hai. Updated code dono dates par valid same pixels ka intersection leta hai.

Example: 2016 values `[10, 100]`, 2022 `[12, missing]`. Independent means 55 aur 12 misleading drop dikhayenge. Paired support sirf first pixel rakhega: difference +2. Test is exact example ko check karta hai.

145 wards save hote hain; har ward ka total pixel count, paired valid count aur coverage fraction visible hai. Mean extreme values se affect hota hai; median/p10/p90 distribution samjhate hain. Across-study mean pixels se calculate hota hai. Har ward ka equal-weight mean lene se tiny aur large wards ko same weight milega, jo different question hai.

Actual paired pixels 591,749 hain. Mean LST 38.62 °C se 46.01 °C, mean difference +7.40 °C hai. Yeh do selected scenes ka surface observation hai. Long-term warming attribution ke liye many dates, season/weather adjustment aur uncertainty analysis chahiye.

## 13. Correlation aur causal claim

Pearson r linear association ka direction/strength describe karta hai. r -0.8 ka matlab “80% temperature kam” nahi. p-value causal probability nahi. Land-cover fractions sum constraints, weather differences, spatial dependence aur model errors association ko affect karte hain.

New exploratory output exact x/y variable names deta hai. NDVI-change vs LST-change r roughly +0.048 (139 wards with >=80% paired LST support) hai. Is defined experiment mein strong negative relationship nahi mila. Yeh har possible vegetation-temperature relationship ko disprove nahi karta.

Model-derived built-up/conversion correlations bhi diagnostic table mein hain, lekin poor class maps ki wajah se reliable causal result nahi. Original r=-0.81 reproduce nahi hua as stated. Convenient number ke liye variable definition badalna ya p<0.001 paste karna avoid karo.

## 14. Python packaging, CLI aur reproducibility

`src/uhi/` package modules ko responsibilities se divide karta hai. `pyproject.toml` dependencies aur `uhi` command entry point define karta hai. Virtual environment dependencies isolate karta hai. `pip install -e .` editable installation hai, source changes reinstall ke bina visible hote hain.

`uhi fetch` pinned dataset files download/verify karta hai. SHA-256 file contents ka fingerprint hai; same size alone integrity proof nahi. Partial download final filename banne se pehle checksum pass karta hai. `uhi sample` bundled small real data par model evaluation karta hai. `uhi run` full analysis karta hai. Analysis command apne aap network nahi chalata.

`results.json` configuration, exact environment versions, date IDs aur metrics save karta hai. `input_manifest.json` exact source hashes deta hai. `predictions_*.csv` answers ke peeche per-example evidence hai. Model metadata feature order/class IDs record karta hai. `joblib` fitted Python model serialize karta hai; trusted artifact hi load karo.

Seed reproducibility mein help karta hai, lekin dataset, split, code aur dependency versions bhi same hone chahiye. Different platform/library versions slight differences la sakte hain. “Production ready” bolne ke liye monitoring, failure handling, deployment/load/security requirements aur independent evaluation alag work hain.

## 15. Tests, Git aur review

pytest tests small known situations mein expected behavior check karte hain. 21 current tests cover nullable IDs, class validation, QA bits, physical scaling, NDVI offset, pixel restoration, spatial grouping/buffer, paired stats aur shifted raster NoData. Tests scientific conclusion true hai yeh prove nahi karte; woh implementation mistakes pakadte hain.

Git branch changes ko separate reviewable history deti hai. Commit ek snapshot hai. Pull request proposed branch aur main ka comparison/review hai. CI workflow dependencies install karke tests aur sample evaluation run karega; local pass aur remote CI pass alag evidence hain. Raw GB-size imagery Git mein nahi, pinned public data source aur hashes use karte hain.

## Actual code padhne ka order

1. `features.py`: pure transformations. Har function ka tiny array example khud calculate karo.
2. `training.py`: source points, band identity checks, required cleanup. Feature rows aur coordinates same source row se kaise linked hain?
3. `evaluation.py`: split → fit → predict → score. Group overlap assertion aur buffer explain karo.
4. `geospatial.py`: reference profile, WarpedVRT, valid zero sentinel. Index aur coordinate ka connection trace karo.
5. `analysis.py`: masked inference aur paired aggregation. Ek 2x2 grid ka manual expected answer likho.
6. `pipeline.py`: functions ko join karta hai. Scientific calculations helper modules mein hain; orchestration result files banata hai.
7. `download.py`, `cli.py`, `pyproject.toml`, tests, workflow: use/run/reproduce/review ka flow.

## Practice complete hone ka evidence

Guide follow karna foundation ke liye enough hai. Interview-ready hone ke liye ye tasks khud karna:

- Notes band karke 90-second project explanation aur five-minute detailed walkthrough do.
- `surface_temperature`, `normalized_difference` aur nullable-ID cleaner independently likho.
- Intentional 300 m grid shift create karke show karo ki direct subtraction kyun unsafe hai.
- Same test points ke 300 m neighbors print karke prove karo ki train set se remove hue.
- Prediction CSV se confusion matrix aur accuracy manually recompute karo.
- Paired-support example ko change karke expected result pehle likho, phir test run karo.
- One change implement karo: e.g. coverage threshold configurable banao, aur meaningful test add karo.
- Explain karo ki poor spatial score ke baad more trees ke bajay new dispersed labels kyun collect karoge.

Koi document interviewer ke saare possible questions guarantee nahi karta. Naye follow-ups ko handle karne ke liye inputs, assumptions, alternatives, failure cases aur evidence explain karna seekho. “Is part ka independent validation abhi pending hai” ek accurate answer hai.
