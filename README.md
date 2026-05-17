# AdaBoost Regressor - Car Price Prediction App

A Machine Learning web application built using **Streamlit** and **AdaBoost Regressor** to predict car prices based on vehicle specifications and performance-related features.

## Live Demo

https://adaboost.streamlit.app/

---

# Project Overview

This project demonstrates a complete **Machine Learning Regression workflow** including:

- Data Collection
- Data Cleaning
- Outlier Detection and Treatment
- Model Training
- Model Evaluation
- Model Deployment using Streamlit

The application predicts estimated car prices using multiple vehicle-related factors.

AdaBoost is an ensemble boosting algorithm that improves prediction accuracy by combining multiple weak learners into a strong predictive model. :contentReference[oaicite:0]{index=0}

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

---

# Machine Learning Algorithm

## AdaBoost Regressor

AdaBoost (Adaptive Boosting) is an ensemble machine learning algorithm used for regression and classification tasks.

The algorithm works by:
- Training multiple weak learners sequentially
- Giving higher importance to previous prediction errors
- Combining weak learners into a strong regressor

Scikit-learn describes AdaBoostRegressor as a meta-estimator that repeatedly fits regressors while adjusting weights toward harder prediction cases. :contentReference[oaicite:1]{index=1}

In this project:
- Input → Car Details
- Output → Predicted Car Price

---

# AdaBoost Formula

Core AdaBoost concept:

:contentReference[oaicite:2]{index=2}

Where:
- \(h_m(x)\) = weak learner
- \(\alpha_m\) = learner weight
- \(M\) = number of estimators

---

# Dataset Information

The dataset contains:

- 5000 rows
- 8 columns

## Features

| Feature | Description |
|---|---|
| CarAge | Age of car |
| Mileage | Distance traveled |
| EngineSize | Engine capacity |
| HorsePower | Engine power |
| FuelEfficiency | Mileage efficiency |
| BrandValue | Brand reputation score |
| OwnerCount | Number of previous owners |
| Price | Target variable |

---

# Project Workflow

## 1. Data Preprocessing

- Removed duplicate rows
- Checked missing values
- Statistical analysis using `describe()`

---

## 2. Outlier Detection using IQR

Outliers were detected using the IQR (Interquartile Range) method.

Formula:

:contentReference[oaicite:3]{index=3}

Lower Bound:

:contentReference[oaicite:4]{index=4}

Upper Bound:

:contentReference[oaicite:5]{index=5}

---

## 3. Outlier Treatment

Detected outliers were treated by replacing extreme values using lower and upper bounds with NumPy operations.

---

## 4. Train-Test Split

Dataset split:
- 80% Training
- 20% Testing

---

## 5. Model Training

Used:

```python
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
