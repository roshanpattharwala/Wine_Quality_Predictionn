# Winery Analytics Platform

Enterprise Machine Learning Web Application for Wine Quality Prediction using Flask, Scikit-learn, and SHAP Explainable AI.

---

# Project Overview

The Winery Analytics Platform is a Flask-based Machine Learning web application developed to predict wine quality using physicochemical characteristics. Multiple machine learning algorithms, including Random Forest, XGBoost, and LightGBM, were developed and evaluated during the model development phase. Based on the comparative evaluation results, the best-performing model was selected and deployed within the Flask application. The platform also integrates SHAP (SHapley Additive exPlanations) to improve model transparency by explaining how each input feature contributes to the prediction.
---

# Features

- Wine Quality Prediction
- Interactive Flask Dashboard
- Random Forest Machine Learning Model
- SHAP Explainable AI
- Model Analytics Dashboard
- Data Visualization
- Feature Importance Analysis
- Responsive User Interface

---

# Technology Stack

## Programming Language

- Python 3.13

## Backend

- Flask

## Machine Learning

- Scikit-learn
- Random Forest
- XGBoost
- LightGBM

## Explainable AI

- SHAP

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Model Storage

- Joblib

---


# Dataset

The project is based on the publicly available Wine Quality datasets from the UCI Machine Learning Repository.

Two separate datasets were used:

- Red Wine Quality Dataset
- White Wine Quality Dataset

Both datasets were combined into a single unified dataset after preprocessing. A new categorical feature named **Wine Type** was introduced to distinguish between red and white wine samples. The merged dataset was then used for feature engineering, exploratory data analysis, model training, evaluation, explainable AI analysis, and deployment.

The final merged dataset contains physicochemical measurements together with the corresponding wine quality score.

Target Variable:

- Wine Quality Score

---

# Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Data Preprocessing
4. Feature Engineering
5. Exploratory Data Analysis
6. Train-Test Split
7. Feature Scaling
8. Model Training
9. Hyperparameter Tuning
10. Model Evaluation
11. SHAP Explainability
12. Flask Deployment

---

# Installation

Move into the project folder

```bash
cd winery_analytics_platform
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

# Running the Application

Run the Flask application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# Dashboard Overview

The dashboard contains:

- Model Information
- Wine Prediction Form
- Prediction Result
- SHAP Explainability
- Analytics Dashboard
- Feature Importance
- Correlation Heatmap
- Model Comparison
- Distribution Analysis

---

# Generated Visualizations

- Correlation Heatmap
- Histograms
- Boxplots
- Violin Plots
- Scatter Plots
- Feature Importance
- SHAP Summary Plot
- SHAP Bar Plot
- SHAP Force Plot
- SHAP Waterfall Plot
- Global Feature Contribution
- Local Feature Contribution
- Confusion Matrix
- Classification Metrics
- Cross Validation Stability
- Prediction Error Distribution

---

# Future Improvements

- Cloud Deployment
- Docker Support
- REST API
- User Authentication
- Deep Learning Models
- Real-Time Prediction Service

---

