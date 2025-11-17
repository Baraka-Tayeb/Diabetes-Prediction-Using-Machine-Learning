# 🩺 Diabetes Prediction Using Machine Learning

A complete end-to-end machine learning pipeline for predicting diabetes using clinical and demographic data.

# 📌 Project Overview

This project builds a full ML workflow—from data loading and exploration to preprocessing, handling imbalance, model training, hyperparameter tuning, and evaluation.
It implements multiple machine learning models (Random Forest, Logistic Regression, Gradient Boosting) and applies advanced techniques such as ADASYN oversampling, GridSearchCV, RandomizedSearchCV, and Optuna for model optimization.

The project aims to identify key factors influencing diabetes and build accurate predictive models.

# 📂 Dataset

File: diabetes_dataset22.csv

Number of rows: Automatically printed in the notebook

Contains demographic, lifestyle, and clinical features such as:

age

gender

smoking_history

bmi

hbA1c_level

blood_glucose_level

hypertension

heart_disease

race categories

Target variable: diabetes (0 = No, 1 = Yes)

# 🧹 Data Preprocessing

The project applies a series of preprocessing steps:

# ✔️ Exploratory Data Analysis (EDA)

-Dataset shape, types, summary statistics

-Missing value inspection

-Outlier detection using IQR + boxplots (Plotly)

-Skewness analysis + histograms

# ✔️ Feature Engineering & Cleaning

-Log-transformation for skewed features (bmi, blood_glucose_level)

-Standardization of numerical features

-One-Hot Encoding for categorical features

-Class imbalance handling using ADASYN

# 🤖 Machine Learning Models

Several ML models are trained and evaluated:

# 1️⃣ Random Forest Classifier

-Baseline model

-RandomizedSearchCV for tuning

-Trained with and without ADASYN

-Feature importance visualization

# 2️⃣ Logistic Regression

-Baseline logistic model

-GridSearchCV for hyperparameter tuning

-Logistic Regression with ADASYN

-Evaluation using:

-Accuracy

-Precision, Recall, F1-score

-ROC-AUC

-Confusion Matrix

# 3️⃣ Gradient Boosting Classifier

-Baseline GBC model

-Hyperparameter tuning using Optuna

-GBC + ADASYN

-GBC + ADASYN + Optuna with cross-validation

# 4️⃣ Overfitting Detection

-Compares train vs test performance

-Alerts if overfitting is detected

# 📊 Evaluation Metrics

-Models are evaluated using:

-Accuracy

-Precision, Recall, F1-Score

-ROC-AUC Score

-Confusion Matrices

-Feature Importance (Random Forest & Gradient Boosting)

# 🧪 Technologies & Libraries

-Python

-NumPy, Pandas

-Matplotlib, Seaborn

-Plotly

-Scikit-Learn

-Imbalanced-Learn (ADASYN)

-Optuna

-ColumnTransformer & Pipelines

# 📈 Key Highlights

- Complete ML lifecycle

- Multiple imbalance-handling strategies

- Advanced hyperparameter tuning

- Strong model comparison

- Visual, interpretable results
