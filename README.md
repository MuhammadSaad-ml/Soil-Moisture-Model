<!-- ================= BADGES ================= -->

<!-- Core Project -->
<img src="https://img.shields.io/badge/Machine_Learning-green">
<img src="https://img.shields.io/badge/Build_with-Python-red">
<img src="https://img.shields.io/badge/Framework-Streamlit-yellow">
<img src="https://img.shields.io/badge/Webapp-Interactive-purple">
<img src="https://img.shields.io/badge/Data-Agriculture-hotpink">
<img src="https://img.shields.io/badge/Smart_Agriculture-AI-brightgreen">
<img src="https://img.shields.io/badge/Precision_Farming-Enabled-success">
<img src="https://img.shields.io/badge/Soil_Health-Monitoring-brown">
<img src="https://img.shields.io/badge/Irrigation-Decision_Support-blue">

<!-- Libraries -->
<img src="https://img.shields.io/badge/Pandas-Data_Analysis-orange">
<img src="https://img.shields.io/badge/NumPy-Scientific_Computing-blue">
<img src="https://img.shields.io/badge/Scikit--Learn-ML_Modeling-f7931e">
<img src="https://img.shields.io/badge/Plotly-Interactive_Charts-3f4f75">

<!-- Models -->
<img src="https://img.shields.io/badge/Regression-Soil_Moisture_Prediction-green">
<img src="https://img.shields.io/badge/Decision_Tree-Regressor-yellowgreen">
<img src="https://img.shields.io/badge/Neural_Network-MLP_Regressor-blue">

<!-- Model Evaluation -->
<img src="https://img.shields.io/badge/Model_Tuning-Interactive-yellow">
<img src="https://img.shields.io/badge/Train_Test_Split-Dynamic-blueviolet">

<!-- Development -->
<img src="https://img.shields.io/badge/Anaconda-Environment-a8b59c">
<img src="https://img.shields.io/badge/VS_Code-IDE-blueviolet">

<!-- Project Quality -->
<img src="https://img.shields.io/badge/Open_Source-Yes-brightgreen">
<img src="https://img.shields.io/badge/End--to--End-ML_Project-gold">
<img src="https://img.shields.io/badge/Beginner_Friendly-Yes-blue">

---

# 🌱 Smart Soil Moisture Prediction Webapp Using Streamlit

This repository contains an interactive **Smart Soil Moisture Prediction Web Application** built using **Streamlit**.

The application predicts **soil moisture (%)** based on environmental and agricultural factors such as:

- Temperature  
- Humidity  
- Rainfall  
- Soil pH  
- Crop type  
- Fertilizer type  
- Region  

The project combines **Exploratory Data Analysis (EDA)**, **Machine Learning**, **model tuning**, and **interactive visualizations** into a single **end-to-end data science web application** for smart agriculture.

---

## ▶ Click On Image To Open Demo
[![Example](https://i.imgur.com/CRstwB3.png)](https://www.youtube.com/)

---

## ✨ Features

### 📊 Dataset Filtering & Exploration
- Filter data by **region**, **crop type**, and **fertilizer type**
- Analyze soil moisture trends under different farming conditions

### 📈 Interactive Data Visualization
- Visualize soil moisture relationships with:
  - Temperature  
  - Humidity  
  - Rainfall  
  - Soil pH  
- Interactive Plotly charts for deeper insights

### 💧 Soil Moisture Classification
Soil moisture values are categorized into:
- 🌵 **Dry**
- 🌾 **Optimal**
- 💧 **Wet**

### 🤖 Machine Learning Models
The application trains and compares:
- 🌳 **Decision Tree Regressor**
- 🤖 **Neural Network (MLP Regressor)**

### ⚙ Model Tuning Controls
Users can dynamically adjust:
- Train/Test split ratio  
- Decision Tree maximum depth  
- Neural Network hidden layer size  

### 📐 Model Evaluation & Metrics
Performance metrics include:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- Standard Deviation of Errors  

Includes **Actual vs Predicted** comparison plots.

### 🔮 Latest Soil Moisture Prediction
- Real-time prediction from both models
- Visual indicator for irrigation decision support

---

## 📁 Project Structure & Files Included

```text
Soil-Moisture-Model/
│
├── .devcontainer/                     # Dev container configuration
│
├── soil_moisture_app.py               # Main Streamlit application
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
│
├── Datasets/
│   ├── Expanded_Mixup_LatLong.csv     # MixUp + Lat/Long augmented dataset
│   ├── Expanded_Soil_Dataset.csv      # Expanded soil dataset
│   ├── augmented_soil_data.csv        # Synthetic soil data
│   ├── Smart_Farming_Crop_Yield_2024.csv
│   └── Smart_Farming_Crop_Yield_Refined_2024.csv
```

---

## 📄 File Description

### `soil_moisture_app.py`
Main Streamlit application file containing:
- Data loading & preprocessing
- Filtering by region, crop & fertilizer
- Interactive Plotly visualizations
- ML model training & evaluation
- Model tuning controls
- Live soil moisture prediction

> Recently **refactored** to improve structure, performance, and readability.

---

### `.devcontainer/`
Provides a **Docker-based development environment**:
- Consistent setup across machines
- Easy onboarding
- Reproducible execution

---

### Datasets
- **Expanded_Mixup_LatLong.csv**  
  MixUp-based augmentation + Lat/Long feature expansion

- **Expanded_Soil_Dataset.csv**  
  Expanded version of original soil data

- **augmented_soil_data.csv**  
  Synthetic data for dataset balancing

- **Smart_Farming_Crop_Yield_2024.csv**  
  Raw crop yield dataset

- **Smart_Farming_Crop_Yield_Refined_2024.csv**  
  Cleaned & refined crop yield data

---

### `requirements.txt`
Includes:
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly
- Statsmodels

---

## 📈 Dataset Expansion & Augmentation

- Original dataset contained ~**500 rows**
- Augmentation techniques used:
  - MixUp sample generation
  - Latitude & Longitude feature engineering
  - Synthetic data creation
  - Dataset merging & refinement

> Result: Improved generalization & reduced overfitting.

---

## 🔄 Data Pipeline Overview

1. Raw soil & crop data collection  
2. Data cleaning & preprocessing  
3. Feature engineering (weather, pH, Lat/Long)  
4. Python-based data augmentation  
5. Dataset validation & expansion  
6. Model training & evaluation via Streamlit

---

## 🕒 Version History (Highlights)

- **Initial Commit** – Base Streamlit soil moisture app  
- **Dataset Expansion** – Added augmented & refined datasets  
- **Refactor Update** – Improved architecture & tuning controls  
- **Dev Container Added** – Simplified environment setup  

---

## ℹ Notes
- Ensure correct dataset path before running the app
- App supports flexible column detection across datasets

---

## 👤 Author

**Muhammad Saad**  
Data Analyst & Data Scientist 🌱
