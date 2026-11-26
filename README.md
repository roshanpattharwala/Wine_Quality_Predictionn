# AI-Powered Predictive Analytics Platform for Wineries

MSc Artificial Intelligence — Capstone Project

## 1. Project overview

This project develops a machine learning classification system to predict wine quality using physicochemical properties of red and white wines.

The project combines both datasets into one dataset and uses machine learning models to classify wine quality scores. It also investigates feature importance and model interpretability using SHAP.

The project is implemented in Python using a Jupyter Notebook.

## 2. Project files

Keep the notebook and both original CSV datasets in the same working directory when running the project.

```text
Wine-Quality-Prediction/
├── README.md
├── Wine_Quality_Prediction.ipynb
├── winequality-red.csv
└── winequality-white.csv
├── app.py
```

## 3. Software requirements

- Python 3
- Jupyter Notebook or Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Joblib
- SciPy
- SHAP

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib scipy shap
```

## 4. How to run the project

1. Open Google Colab.
2. Upload Wine_Quality_Prediction.ipynb and open it.
3. Upload winequality-red.csv and winequality-white.csv to the current runtime's working directory.
4. Make sure both CSV filenames match the names expected by the notebook.
5. Select Runtime > Run all to execute the notebook in order.
6. Allow the data preprocessing, model training, hyperparameter optimisation and evaluation cells to finish.
7. Review the experimental results, classification metrics, confusion matrix and SHAP visualisations.
8. Run the model-export cells to generate the trained model and supporting output files.

## 5. Project workflow

The notebook follows these main steps:

1. **Data ingestion and merging** – Loads the red and white wine datasets and combines them into one dataset.
2. **Data understanding** – Examines data structure, missing values, duplicates and statistical summaries.
3. **Class imbalance analysis** – Investigates the distribution of wine quality scores.
4. **Exploratory data analysis** – Uses visualisations and correlation analysis to explore the dataset.
5. **Outlier analysis** – Identifies and examines outliers in the data.
6. **Feature engineering** – Creates `Free_to_Total_SO2_Ratio` and `Total_Acidity_Index`.
7. **Feature and target preparation** – Separates the input features from the wine quality target.
8. **Train-test splitting** – Uses a grouped, stratified 80:20 split to keep identical predictor patterns together.
9. **Preprocessing and validation** – Applies IQR-based winsorisation and standardisation within a machine learning pipeline, with five-fold grouped cross-validation.
10. **Model training and comparison** – Compares Random Forest, XGBoost and LightGBM against a majority-class baseline using model evaluation metrics and hyperparameter tuning.
11. **Error analysis** – Examines prediction errors and near-miss quality predictions.
12. **SHAP explainability** – Uses global and local SHAP explanations to interpret the selected model.
13. **Model saving** – Saves the selected model pipeline and supporting deployment information.

## 6. Model evaluation

The project evaluates model performance using:

- Accuracy
- Weighted F1-score
- Macro F1-score
- Minority-class F1-score
- Classification report
- Confusion matrix
- Near-miss and ordinal error analysis

The final model is selected based on cross-validation weighted F1-score.
