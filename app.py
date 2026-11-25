# =============================================================================
# WINERY ANALYTICS PLATFORM
# Flask Deployment Application
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    flash
)

# =============================================================================
# CREATE FLASK APPLICATION
# =============================================================================

app = Flask(__name__)

app.secret_key = "winery_secret_key_2026"

# =============================================================================
# PROJECT DIRECTORIES
# =============================================================================

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# =============================================================================
# MODEL FILES
# =============================================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_winery_model.pkl"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

LABEL_ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)

DEPLOYMENT_FILE = os.path.join(
    MODEL_DIR,
    "deployment_info.pkl"
)

# =============================================================================
# LOAD DEPLOYMENT FILES
# =============================================================================

try:

    best_model = joblib.load(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    feature_names = joblib.load(
        FEATURE_FILE
    )

    deployment_info = joblib.load(
        DEPLOYMENT_FILE
    )

    if os.path.exists(
        LABEL_ENCODER_FILE
    ):

        label_encoder = joblib.load(
            LABEL_ENCODER_FILE
        )

    else:

        label_encoder = None

except Exception as error:

    raise RuntimeError(

        f"\nDeployment loading failed.\n{error}"

    )

# =============================================================================
# MODEL INFORMATION
# =============================================================================

MODEL_NAME = deployment_info["Model Name"]

MODEL_TYPE = deployment_info["Model Type"]

CV_SCORE = deployment_info["Cross Validation F1"]

FEATURE_COUNT = deployment_info["Number of Features"]

TRAINING_SAMPLES = deployment_info["Training Samples"]

TESTING_SAMPLES = deployment_info["Testing Samples"]

FEATURE_ORDER = feature_names.copy()

# =============================================================================
# APPLICATION STARTUP MESSAGE
# =============================================================================

print("\n")

print("=" * 70)

print("WINERY ANALYTICS PLATFORM")

print("=" * 70)

print(f"Model Loaded        : {MODEL_NAME}")

print(f"Model Type          : {MODEL_TYPE}")

print(f"Training Samples    : {TRAINING_SAMPLES}")

print(f"Testing Samples     : {TESTING_SAMPLES}")

print(f"Number of Features  : {FEATURE_COUNT}")

print(f"Cross Validation F1 : {CV_SCORE}")

print("=" * 70)

print("Deployment Ready")

print("=" * 70)

print("\n")

# =============================================================================
# HELPER FUNCTION
# Validate Numeric Input
# =============================================================================

def get_float(form, field_name):

    value = form.get(field_name, "").strip()

    if value == "":

        raise ValueError(f"{field_name} is required.")

    try:

        value = float(value)

    except ValueError:

        raise ValueError(f"{field_name} must be numeric.")

    if value < 0:

        raise ValueError(f"{field_name} cannot be negative.")

    return value


# =============================================================================
# HELPER FUNCTION
# Prepare Feature Vector
# =============================================================================

def prepare_features(form):

    # -------------------------------------------------------------------------
    # Read User Inputs
    # -------------------------------------------------------------------------

    fixed_acidity = get_float(form, "fixed_acidity")

    volatile_acidity = get_float(form, "volatile_acidity")

    citric_acid = get_float(form, "citric_acid")

    residual_sugar = get_float(form, "residual_sugar")

    chlorides = get_float(form, "chlorides")

    free_sulfur_dioxide = get_float(form, "free_sulfur_dioxide")

    total_sulfur_dioxide = get_float(form, "total_sulfur_dioxide")

    density = get_float(form, "density")

    ph = get_float(form, "ph")

    sulphates = get_float(form, "sulphates")

    alcohol = get_float(form, "alcohol")

    # -------------------------------------------------------------------------
    # Wine Type
    # -------------------------------------------------------------------------

    wine_type = form.get("wine_type")

    if wine_type == "red":

        is_red = 1

    elif wine_type == "white":

        is_red = 0

    else:

        raise ValueError("Please select a valid Wine Type.")

    # -------------------------------------------------------------------------
    # Engineered Feature 1
    # Free_to_Total_SO2_Ratio
    # -------------------------------------------------------------------------

    if total_sulfur_dioxide == 0:

        free_to_total_ratio = 0

    else:

        free_to_total_ratio = (

            free_sulfur_dioxide /
            total_sulfur_dioxide

        )

    # -------------------------------------------------------------------------
    # Engineered Feature 2
    # Total_Acidity_Index
    # -------------------------------------------------------------------------

    total_acidity_index = (

        fixed_acidity +
        volatile_acidity -
        ph

    )

    # -------------------------------------------------------------------------
    # Feature Dictionary
    # (Must Match feature_names.pkl Exactly)
    # -------------------------------------------------------------------------

    feature_dictionary = {

        "fixed acidity": fixed_acidity,

        "volatile acidity": volatile_acidity,

        "citric acid": citric_acid,

        "residual sugar": residual_sugar,

        "chlorides": chlorides,

        "free sulfur dioxide": free_sulfur_dioxide,

        "total sulfur dioxide": total_sulfur_dioxide,

        "density": density,

        "pH": ph,

        "sulphates": sulphates,

        "alcohol": alcohol,

        "is_red": is_red,

        "Free_to_Total_SO2_Ratio": free_to_total_ratio,

        "Total_Acidity_Index": total_acidity_index

    }

    # -------------------------------------------------------------------------
    # Arrange Features in Training Order
    # -------------------------------------------------------------------------

    ordered_features = [

        feature_dictionary[name]

        for name in FEATURE_ORDER

    ]

    # -------------------------------------------------------------------------
    # Create DataFrame
    # -------------------------------------------------------------------------

    input_df = pd.DataFrame(

        [ordered_features],

        columns=FEATURE_ORDER

    )

    # -------------------------------------------------------------------------
    # Standard Scaling
    # -------------------------------------------------------------------------

    scaled_features = scaler.transform(

        input_df

    )

    return scaled_features

# =============================================================================
# HOME PAGE
# =============================================================================

@app.route("/")

def home():

    return render_template(

        "dashboard.html",

        model_name=MODEL_NAME,

        model_type=MODEL_TYPE,

        feature_count=FEATURE_COUNT,

        cv_score=CV_SCORE,

        prediction=None,

        probability=None,

        prediction_status=None

    )

# =============================================================================
# PREDICTION ROUTE
# =============================================================================

@app.route("/predict", methods=["POST"])

def predict():

    try:

        # ---------------------------------------------------------------------
        # Prepare Features
        # ---------------------------------------------------------------------

        scaled_features = prepare_features(

            request.form

        )

        # ---------------------------------------------------------------------
        # Predict Wine Quality
        # ---------------------------------------------------------------------

        prediction = best_model.predict(

            scaled_features

        )[0]

        # ---------------------------------------------------------------------
        # Decode Label (If Required)
        # ---------------------------------------------------------------------

        if label_encoder is not None:

            prediction = label_encoder.inverse_transform(

                [int(prediction)]

            )[0]

        # ---------------------------------------------------------------------
        # Prediction Probability
        # ---------------------------------------------------------------------

        probability = "Not Available"

        if hasattr(best_model, "predict_proba"):

            probabilities = best_model.predict_proba(

                scaled_features

            )

            confidence = np.max(

                probabilities

            )

            probability = f"{confidence*100:.2f}%"

        # ---------------------------------------------------------------------
        # Prediction Status
        # ---------------------------------------------------------------------

        prediction_status = "Prediction Completed Successfully"

        # ---------------------------------------------------------------------
        # Render Dashboard
        # ---------------------------------------------------------------------

        return render_template(

            "dashboard.html",

            model_name=MODEL_NAME,

            model_type=MODEL_TYPE,

            feature_count=FEATURE_COUNT,

            cv_score=CV_SCORE,

            prediction=prediction,

            probability=probability,

            prediction_status=prediction_status

        )

    except Exception as error:

        flash(

            str(error),

            "error"

        )

        return render_template(

            "dashboard.html",

            model_name=MODEL_NAME,

            model_type=MODEL_TYPE,

            feature_count=FEATURE_COUNT,

            cv_score=CV_SCORE,

            prediction=None,

            probability=None,

            prediction_status=None

        )
    
    # =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )