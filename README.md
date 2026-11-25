from pathlib import Path

readme = r"""# AI-Powered Predictive Analytics Platform for Wineries

**MSc AI Capstone Project**  
**Author:** Roshan Chiragkumar Pattharwala  
**Student ID:** A00062620

## Project overview

This project develops a machine-learning workflow to predict wine-quality scores from physicochemical measurements. It combines the red-wine and white-wine datasets, explores and prepares the data, compares several classification models, evaluates prediction errors, explains the selected model with SHAP, and saves the trained model and supporting deployment artifacts.

The target variable is `quality`. The notebook treats this as a multiclass classification problem and retains the original wine-quality labels in its reported predictions.

## Project objectives

- Combine red and white wine records into one dataset while retaining a wine-type indicator.
- Audit the data and explore feature distributions, correlations, class imbalance and potential outliers.
- Create additional physicochemical features.
- Prepare a train/test split that keeps identical predictor records together.
- Use leakage-conscious preprocessing and grouped, stratified cross-validation.
- Compare a majority-class baseline with Random Forest, XGBoost and LightGBM.
- Evaluate the selected model using classification metrics and ordinal/near-miss error analysis.
- Interpret model behaviour using global and local SHAP explanations.
- Save the complete fitted pipeline and supporting files for later use.

## Repository contents

The notebook expects the following dataset files to be available:

```text
winequality-red.csv
winequality-white.csv
```

The main project file is:

```text
Wine_Quality_Prediction(8).ipynb
```

The notebook also creates output files when its relevant cells are executed. These include:

| Output | Purpose |
|---|---|
| `model_comparison_cv_and_test.csv` | Cross-validation and held-out test comparison of the baseline and candidate models |
| `minority_class_test_metrics.csv` | Held-out test metrics for minority quality classes |
| `selected_model_classification_report.csv` | Classification report for the selected model |
| `selected_model_confusion_matrix.png` | Confusion-matrix visualisation |
| `near_miss_summary.csv` | Summary of exact and near-miss predictions and mean absolute error |
| `near_miss_error_distribution.csv` | Distribution of absolute quality-prediction errors |
| `shap_global_feature_importance.png` | Global SHAP feature-importance plot |
| `shap_local_force.png` | Local SHAP explanation for a held-out observation |
| `engineered_features_distribution.png` | Distributions of the engineered features |
| `best_winery_model.pkl` | Complete fitted preprocessing-and-classification pipeline |
| `scaler.pkl` | Fitted feature scaler |
| `winsorizer.pkl` | Fitted IQR winsorisation transformer |
| `label_encoder.pkl` | Target-label encoder used for XGBoost |
| `feature_names.pkl` | Feature names and expected feature order |
| `deployment_info.pkl` | Model and evaluation metadata |

These are generated outputs; they may not exist until the notebook has run successfully.

## Dataset and input requirements

Place both CSV files in the same working directory from which the notebook is run:

- `winequality-red.csv`
- `winequality-white.csv`

The notebook reads each file with a semicolon (`;`) delimiter. Each dataset must contain the wine-quality target column named `quality` and the physicochemical predictor columns expected by the notebook.

During ingestion, the notebook adds an `is_red` column:

- `1` identifies a record from the red-wine dataset.
- `0` identifies a record from the white-wine dataset.

The two datasets are then concatenated row-wise. The notebook does not download the datasets automatically, so they must be supplied separately.

## Methodology and workflow

The notebook is organised into the following stages.

1. **Environment setup** – checks and installs/imports the Python packages used by the workflow.
2. **Data ingestion and merging** – loads the two CSV files, adds the wine-origin indicator and combines the records.
3. **Data understanding and audit** – examines dataset structure, data types, missing values, duplicates and descriptive statistics.
4. **Target-class profiling** – examines the distribution of wine-quality labels.
5. **Exploratory data analysis** – uses correlation analysis, histograms, boxplots, violin plots and scatter plots.
6. **Outlier profiling** – reports IQR-based outlier information for selected measurements.
7. **Feature engineering** – creates `Free_to_Total_SO2_Ratio` and `Total_Acidity_Index`.
8. **Feature and target preparation** – separates predictors from `quality` and groups rows with identical predictor values.
9. **Train/test splitting** – uses a five-fold `StratifiedGroupKFold` split to form the training and held-out test partitions while keeping duplicate predictor groups together.
10. **Preprocessing and validation** – places IQR winsorisation and standardisation inside a scikit-learn pipeline and uses grouped, stratified cross-validation.
11. **Model training and comparison** – evaluates a majority-class baseline and tunes Random Forest, XGBoost and LightGBM using `RandomizedSearchCV`.
12. **Error analysis** – examines exact predictions, predictions within one or two quality points, and larger errors.
13. **Explainability** – calculates global and local SHAP explanations for the selected model.
14. **Model and artifact export** – saves the complete fitted pipeline and supporting deployment files.

### Engineered features

The notebook creates two additional predictors:

- `Free_to_Total_SO2_Ratio` – the ratio of free sulfur dioxide to total sulfur dioxide.
- `Total_Acidity_Index` – calculated as `(fixed acidity + volatile acidity) / pH`.

### Validation and model selection

The notebook uses grouped, stratified five-fold cross-validation. The outer split is used to reserve a held-out test set. The inner grouped folds are used for baseline evaluation and model tuning.

The candidate classifiers are:

- Random Forest
- XGBoost
- LightGBM

Hyperparameter search uses `RandomizedSearchCV` with 15 sampled configurations per candidate model. The search refits using weighted F1. The notebook also reports accuracy, weighted F1, macro F1 and minority-class F1, alongside held-out test results.

The notebook's `RANDOM_STATE` variable controls reproducibility where it is supplied to the splitting and model procedures.

## Software requirements

The notebook uses Python and the following packages:

- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- LightGBM
- SciPy
- SHAP
- joblib

The first setup cell checks for the required packages and attempts to install missing packages. A Python environment with permission to install packages is therefore useful. Jupyter Notebook or JupyterLab is required to run the `.ipynb` file.

## How to run

1. Clone or download this project repository.
2. Make sure the two wine CSV files are present in the notebook's working directory.
3. Create and activate a Python environment if required.
4. Install Jupyter Notebook or JupyterLab.
5. Open `Wine_Quality_Prediction(8).ipynb`.
6. Run the notebook cells in order, from the environment setup and data ingestion through to model export.
7. Review the generated evaluation tables, plots and saved model artifacts.

For example, after installing Jupyter in your environment:

```bash
jupyter notebook
```

Then open the notebook from the Jupyter file browser.

## Using the saved model

The principal deployment artifact is:

```text
best_winery_model.pkl
```

It contains the complete fitted scikit-learn pipeline, including the fitted preprocessing steps and selected classifier. The notebook explicitly saves this full pipeline so that inference can be performed using raw feature values in the expected feature schema and order.

A basic loading example is:

```python
import joblib
import pandas as pd

model = joblib.load("best_winery_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Replace these example values with one real observation.
new_wine = pd.DataFrame(
    [[/* values in the required feature order */]],
    columns=feature_names
)

predicted_quality = model.predict(new_wine)
print(predicted_quality)
```

**Important:** Replace the placeholder with real numeric values for every feature listed in `feature_names.pkl`. The input must include the engineered features and `is_red`, and must use the same feature names and order as the training data. Do not pass only the original physicochemical columns if the saved pipeline was trained with the additional features.

For XGBoost, the notebook maps encoded predictions back to the original wine-quality labels during evaluation. The saved complete pipeline contains the fitted classifier; consult the saved `label_encoder.pkl` and notebook prediction logic when implementing a separate inference interface.

## Reproducibility and interpretation notes

- Run the notebook from top to bottom so that the required variables and fitted objects are created in sequence.
- Keep the input CSV files unchanged when reproducing the reported results.
- The split and model procedures use a configured random state where specified in the notebook.
- Preprocessing transformations are fitted within the model pipeline and cross-validation process rather than being fitted once on the complete dataset.
- Identical predictor patterns are grouped so they do not cross the train/test boundary or grouped validation folds.
- SHAP values explain the selected model's outputs; they do not demonstrate that a feature causes a change in wine quality.
- Actual performance figures should be taken from the output produced by running the notebook. This README does not assume or invent model scores.

## Limitations

- The model learns from the supplied wine datasets; its performance on wines from other producers, regions, years or measurement processes is not established by this notebook alone.
- Wine quality is represented by the dataset's discrete quality score, so predictions should be interpreted as estimates of those labels rather than a complete assessment of sensory quality.
- The notebook is an experimental modelling workflow. A production deployment would require additional input validation, monitoring, versioning and testing in the intended operating environment.

## Author

**Roshan Chiragkumar Pattharwala**  
MSc Artificial Intelligence – Capstone Project
"""

out = Path("/mnt/data/README.md")
out.write_text(readme, encoding="utf-8")
print(f"Created {out} ({out.stat().st_size:,} bytes)")
