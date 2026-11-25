# Wine Quality Prediction: An AI-Powered Predictive Analytics Platform for Wineries

MSc Artificial Intelligence — Capstone Project  
University of Roehampton

## 1. Project overview

This project develops a machine-learning-based predictive analytics platform to estimate wine quality using physicochemical properties of red and white wines.

The project combines red-wine and white-wine datasets and investigates how chemical measurements, feature engineering and different machine-learning algorithms can be used to predict wine-quality scores.

The modelling workflow uses Python and several machine-learning libraries, including Scikit-learn, XGBoost and LightGBM. It includes exploratory data analysis, data preprocessing, feature engineering, model optimisation, performance evaluation and model explainability using SHAP.

The Jupyter notebook is the main research artefact. It contains the complete data analysis, model development, experimental evaluation and model-export workflow.

## 2. Project files

Keep the notebook and the two original input datasets in the same working directory when running the project.

```text
Wine-Quality-Predictionn/
│
├── README.md
├── Wine_Quality_Prediction.ipynb
├── winequality-red.csv
├── winequality-white.csv
└── app.py

```

The notebook generates the experimental results, visualisations and model artefacts when the relevant cells are executed successfully. These generated files may not exist before running the notebook.

### Input datasets

| File | Description |
|---|---|
| `winequality-red.csv` | Physicochemical measurements and quality scores for red wine |
| `winequality-white.csv` | Physicochemical measurements and quality scores for white wine |

Both datasets use a semicolon (`;`) delimiter.

The notebook loads the datasets, adds a wine-type indicator and combines the records into a single dataset.

The `is_red` feature identifies the wine type:

- `1` represents red wine.
- `0` represents white wine.

The target variable is `quality`, which represents the wine-quality score.

## 3. Software requirements

- Python 3
- Jupyter Notebook, JupyterLab or Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- SciPy
- SHAP
- Joblib

The notebook includes an environment setup section that checks and installs the required Python packages.

For a local Python environment, the dependencies can be installed using:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm scipy shap joblib notebook
```

## 4. How to run the project

### Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Wine_Quality_Prediction.ipynb` and open it.
3. Upload `winequality-red.csv` and `winequality-white.csv` to the current runtime's working directory.
4. Make sure both CSV filenames match the names expected by the notebook.
5. Select **Runtime > Run all** to execute the notebook in order.
6. Allow the data preprocessing, model training, hyperparameter optimisation and evaluation cells to finish.
7. Review the experimental results, classification metrics, confusion matrix and SHAP visualisations.
8. Run the model-export cells to generate the trained model and supporting output files.
9. Download the generated CSV files, plots and model artefacts from the Colab Files panel if required.

### Local Jupyter Notebook

Install the dependencies listed in Section 3, place the notebook and both CSV datasets in the same project directory, and open a terminal in that directory.

Run:

```bash
jupyter notebook
```

Open `Wine_Quality_Prediction.ipynb` and execute the cells from top to bottom.

The original CSV datasets must be accessible using the relative filenames expected by the notebook.

## 5. Model architecture and methodology

The project follows a structured machine-learning workflow consisting of data ingestion, exploratory analysis, feature engineering, preprocessing, model training, hyperparameter optimisation, evaluation and explainability.

### Data preparation

The red-wine and white-wine datasets are combined into a unified dataset.

The data preparation process includes:

- Dataset inspection and structural analysis.
- Data-type verification.
- Missing-value and duplicate checks.
- Descriptive statistical analysis.
- Quality-score distribution analysis.
- Feature correlation analysis.
- Outlier profiling using the Interquartile Range (IQR) method.

### Feature engineering

Two additional physicochemical features are created to improve the information available to the machine-learning models.

| Feature | Description |
|---|---|
| `Free_to_Total_SO2_Ratio` | Ratio of free sulfur dioxide to total sulfur dioxide |
| `Total_Acidity_Index` | Combined acidity measure calculated using fixed acidity, volatile acidity and pH |

The wine-type indicator `is_red` is also retained as a predictor.

### Data preprocessing

The notebook applies a structured preprocessing workflow involving:

- Feature and target separation.
- Group-aware train/test splitting.
- IQR-based winsorisation.
- Standardisation using a fitted scaler.
- Grouped and stratified cross-validation.

The notebook uses `StratifiedGroupKFold` to help maintain quality-class representation while preventing identical predictor records from crossing training and validation boundaries.

Preprocessing transformations are incorporated into the machine-learning pipeline to reduce data leakage during model evaluation and hyperparameter optimisation.

### Machine-learning algorithms

The project compares a majority-class baseline with three machine-learning classifiers.

| Model | Description |
|---|---|
| Majority-Class Baseline | Provides a reference performance level using the most frequent quality class |
| Random Forest | Ensemble learning algorithm based on multiple decision trees |
| XGBoost | Gradient-boosting algorithm designed for efficient predictive modelling |
| LightGBM | Gradient-boosting framework designed for efficient tree-based learning |

The candidate models are evaluated using grouped and stratified cross-validation.

Hyperparameter optimisation is performed using `RandomizedSearchCV`, with 15 sampled configurations per candidate model.

Weighted F1 is used as the model-selection scoring metric.

## 6. Experimental configuration

The experimental workflow uses the following configuration:

| Parameter | Configuration |
|---|---|
| Problem type | Multiclass classification |
| Target variable | `quality` |
| Dataset | Combined red-wine and white-wine datasets |
| Wine-type indicator | `is_red` |
| Cross-validation | 5-fold StratifiedGroupKFold |
| Data splitting | Group-aware train/test splitting |
| Preprocessing | IQR winsorisation and standardisation |
| Baseline model | Majority-class classifier |
| Candidate models | Random Forest, XGBoost and LightGBM |
| Hyperparameter optimisation | RandomizedSearchCV |
| Search configurations | 15 per candidate model |
| Model-selection metric | Weighted F1-score |
| Explainability | SHAP |

The notebook uses a configured random state where specified to support reproducibility.

## 7. Model evaluation and experimental results

The project evaluates the performance of the candidate machine-learning models using cross-validation and held-out test data.

The evaluation process considers multiple classification metrics rather than relying exclusively on accuracy.

### Evaluation metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified observations |
| Weighted F1-score | Evaluates the balance between precision and recall while accounting for class support |
| Macro F1-score | Calculates the average F1-score across quality classes |
| Minority-Class F1 | Examines model performance on less frequently represented quality classes |
| Confusion Matrix | Identifies correct and incorrect predictions across quality classes |

### Error analysis

The notebook also performs additional prediction-error analysis.

This includes:

- Exact quality-score prediction.
- Predictions within one quality point.
- Predictions within two quality points.
- Larger prediction errors.
- Mean Absolute Error.
- Distribution of absolute prediction errors.

This analysis provides additional insight into how close the predicted quality scores are to the actual quality labels.

### Model explainability

SHAP (SHapley Additive exPlanations) is used to interpret the selected model's predictions.

The explainability analysis includes:

- Global feature importance.
- Local prediction explanations.
- Feature contribution analysis.
- Identification of influential physicochemical measurements.

The SHAP visualisations help explain which features contribute most strongly to the model's predictions.

Actual performance figures should be taken from the results generated by executing the notebook.

## 8. Generated output files

The notebook generates several files to support model evaluation, interpretation and deployment.

| Generated file | Description |
|---|---|
| `model_comparison_cv_and_test.csv` | Comparison of baseline and candidate models using cross-validation and held-out test results |
| `minority_class_test_metrics.csv` | Evaluation metrics for minority quality classes |
| `selected_model_classification_report.csv` | Classification report for the selected model |
| `selected_model_confusion_matrix.png` | Visual representation of classification performance |
| `near_miss_summary.csv` | Summary of exact and near-miss predictions and prediction errors |
| `near_miss_error_distribution.csv` | Distribution of absolute prediction errors |
| `shap_global_feature_importance.png` | Global SHAP feature-importance visualisation |
| `shap_local_force.png` | Local SHAP explanation for a held-out observation |
| `engineered_features_distribution.png` | Distribution plots for engineered features |
| `best_winery_model.pkl` | Complete fitted preprocessing and classification pipeline |
| `scaler.pkl` | Fitted feature standardisation transformer |
| `winsorizer.pkl` | Fitted IQR winsorisation transformer |
| `label_encoder.pkl` | Target-label encoder used for XGBoost |
| `feature_names.pkl` | Feature names and expected feature order |
| `deployment_info.pkl` | Model and evaluation metadata |

These files provide the outputs required for reviewing the experimental workflow and reusing the trained model.

## 9. Using the saved model

The principal deployment artefact is:

```text
best_winery_model.pkl
```

It contains the complete fitted machine-learning pipeline, including preprocessing transformations and the selected classifier.

The model can be loaded using Joblib:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("best_winery_model.pkl")

# Load the expected feature names
feature_names = joblib.load("feature_names.pkl")

# Prepare new wine data
new_wine = pd.DataFrame(
    [[/* Insert feature values in the required order */]],
    columns=feature_names
)

# Predict wine quality
prediction = model.predict(new_wine)

print("Predicted wine quality:", prediction)
```

The example requires real numeric values for all expected features.

The input data must contain the original physicochemical measurements, engineered features and wine-type indicator in the expected feature schema and order.

For an independent deployment interface, the prediction handling should also follow the target-label encoding used during training.

## 10. Limitations and intended use

This project is an academic machine-learning research implementation designed to investigate wine-quality prediction using physicochemical measurements.

The model learns patterns from the supplied red-wine and white-wine datasets. Its performance on wines from different producers, geographical regions, production years or measurement processes has not been independently established.

Wine quality is represented using discrete quality scores. Therefore, the predictions should be interpreted as estimates of the dataset's quality labels rather than a complete assessment of sensory characteristics.

Other limitations include:

- Class imbalance between different wine-quality categories.
- Potential difficulty predicting underrepresented quality classes.
- Dependence on the quality and representativeness of the input datasets.
- Limited evidence of generalisation to external wine datasets.
- The need for additional validation before real-world deployment.

SHAP explanations identify model-feature contributions but do not establish causal relationships between physicochemical measurements and wine quality.

The notebook and generated results are intended for academic assessment, experimental analysis and reproducibility.

---

**Author:** Roshan Chiragkumar Pattharwala  
**Programme:** MSc Artificial Intelligence  
**Institution:** University of Roehampton
