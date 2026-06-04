# Urban Heat Island (UHI) Analysis – Geospatial & ML Study

## 📌 Project Overview
A comprehensive 7-year geospatial analysis of the Urban Heat Island (UHI) effect across 150 urban wards in Hyderabad. This project integrates spatial data processing with Machine Learning to evaluate climate trends and quantify the impact of land-use conversion on surface temperatures. 

## 🚀 Key Metrics & Results
* **Temperature Anomaly:** Quantified a city-wide minimum surface temperature rise of **+8.2°C**.
* **Root Cause Identification:** Pinpointed bare-land-to-urban conversion as the primary driver of temperature increase.
* **Statistical Significance:** Achieved high statistical confidence with a Pearson correlation of **r = -0.81** (p < 0.001).

## 🛠️ Tech Stack & Tools
* **Languages:** Python
* **Machine Learning:** Scikit-learn (Random Forest Regressor)
* **Geospatial Tools:** QGIS, GDAL
* **Data Processing:** Pandas, NumPy
* **Dataset:** Landsat 8 Satellite Imagery (Temporal data covering 7 years)

## 📂 Dataset Access
Due to the large size of the raw Landsat 8 satellite imagery (8 GB), the full dataset is hosted externally. 
* [Link to Full Dataset on Google Drive/Kaggle] *(Insert your link here)*
* A small pre-processed sample dataset is available in the `/data` directory for code testing.

## ⚙️ Methodology
1. **Data Ingestion:** Acquired and pre-processed 7 years of Landsat 8 imagery.
2. **Geospatial Processing (QGIS):** Mapped temporal spatial variance across 150 municipal wards.
3. **Model Training:** Engineered a data processing pipeline to train and optimize a Random Forest model on the spatial variables to extract predictive climate trends.
