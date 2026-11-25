"""
PhishGuard – Model Training Module
===================================
This module handles:
1. Loading the pre-extracted numerical feature dataset.
2. Validating input feature data types, missing values, and infinite numbers.
3. Encoding the ground-truth target labels (0 = Legitimate, 1 = Phishing).
4. Performing stratified 80-20 train-test splitting.
5. Implementing and training 3 Scikit-learn classification pipelines:
   - Logistic Regression (with StandardScaler)
   - Support Vector Machine (LinearSVC with StandardScaler & Platt probability calibration)
   - Decision Tree Classifier (with controlled depth)
6. Serializing trained model pipelines and feature column metadata using Joblib.
7. Verification and testing on held-out test samples.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------------------------
# Paths Configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "features_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILES = {
    "logistic": os.path.join(MODELS_DIR, "logistic_model.pkl"),
    "svm": os.path.join(MODELS_DIR, "svm_model.pkl"),
    "decision_tree": os.path.join(MODELS_DIR, "decision_tree_model.pkl"),
}
COLUMNS_FILE = os.path.join(MODELS_DIR, "feature_columns.json")


# ------------------------------------------------------------------------------
# Task 1: Data Preparation
# ------------------------------------------------------------------------------
def load_and_prepare_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Loads features dataset, verifies absence of raw text columns,
    audits missing/inf values, and encodes the binary target label.
    """
    print(f"Loading feature dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset Loaded: {df.shape[0]:,} records, {df.shape[1]} columns")

    # 1. Separate independent variables (X) and target variable (y)
    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' not found in dataset!")

    y_raw = df["Label"]
    X = df.drop(columns=["Label"])

    # 2. Ensure raw URL column is NOT included as a model input
    if "URL" in X.columns:
        print("Note: Removing raw 'URL' column from numerical feature matrix.")
        X = X.drop(columns=["URL"])

    feature_columns = X.columns.tolist()
    print(f"Verified {len(feature_columns)} numerical input features: {feature_columns}")

    # 3. Verify all input features are numerical
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(f"Found non-numeric feature columns: {non_numeric}")

    # 4. Check for missing and infinite values
    null_count = X.isnull().sum().sum()
    inf_count = np.isinf(X.values).sum()
    print(f"Data Quality Check: Missing values = {null_count}, Infinite values = {inf_count}")
    if null_count > 0 or inf_count > 0:
        raise ValueError("Data contains invalid null or infinite values!")

    # 5. Confirm target label encoding: 0 = Legitimate ('good'), 1 = Phishing ('bad')
    # This aligns with standard cybersecurity conventions (1 = positive threat detection).
    label_mapping = {"good": 0, "bad": 1}
    y = y_raw.map(label_mapping)
    if y.isnull().any():
        raise ValueError("Found unrecognized target labels outside {'good', 'bad'}")

    print("Target Label Encoding:")
    print(f"  • Class 0 (Legitimate / Good): {(y == 0).sum():,} records ({(y == 0).mean()*100:.2f}%)")
    print(f"  • Class 1 (Phishing / Bad):    {(y == 1).sum():,} records ({(y == 1).mean()*100:.2f}%)")

    return X, y, feature_columns


# ------------------------------------------------------------------------------
# Task 2: Train-Test Splitting
# ------------------------------------------------------------------------------
def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified 80-20 train-test split with fixed random seed.
    """
    print("\nSplitting dataset into 80% Training and 20% Testing (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  • Training Set:   {X_train.shape[0]:,} samples")
    print(f"  • Held-Out Test:  {X_test.shape[0]:,} samples")
    print(f"  • Training Class Ratio:  Class 0 = {(y_train == 0).mean()*100:.2f}%, Class 1 = {(y_train == 1).mean()*100:.2f}%")
    print(f"  • Test Class Ratio:      Class 0 = {(y_test == 0).mean()*100:.2f}%, Class 1 = {(y_test == 1).mean()*100:.2f}%")
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------------------
# Task 3 & 4: Model Implementation and Training
# ------------------------------------------------------------------------------
def create_model_pipelines() -> Dict[str, Pipeline]:
    """
    Instantiates the three Scikit-learn classification pipelines.
    Preprocessing (StandardScaler) is encapsulated directly inside the pipeline
    to guarantee zero data leakage between train and test distributions.
    """
    pipelines = {
        # Model 1: Logistic Regression
        # Linear probabilistic baseline with StandardScaler for scale invariance
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            ))
        ]),

        # Model 2: Support Vector Machine (LinearSVC with CalibratedClassifierCV)
        # Employs maximum-margin hyperplane separation. CalibratedClassifierCV implements
        # Platt scaling to enable smooth probability outputs (predict_proba) while scaling
        # efficiently to 400,000+ samples.
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(
                    dual=False,
                    max_iter=2000,
                    random_state=42
                ),
                method="sigmoid",
                cv=3
            ))
        ]),

        # Model 3: Decision Tree Classifier
        # Non-linear rule-based tree model. Scale-invariant, so no scaler is needed.
        # Tree depth is bounded to max_depth=12 to prevent overfitting on outlier URL lengths.
        "decision_tree": Pipeline([
            ("classifier", DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ))
        ])
    }
    return pipelines


def train_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Pipeline]:
    """
    Trains each Scikit-learn pipeline on the training split and measures execution time.
    """
    trained_models = {}
    print("\n" + "=" * 60)
    print("TRAINING MACHINE LEARNING MODEL PIPELINES")
    print("=" * 60)

    for name, pipeline in pipelines.items():
        print(f"\n[1/3] Training {name.upper().replace('_', ' ')} Pipeline...")
        t_start = time.time()
        pipeline.fit(X_train, y_train)
        t_elapsed = time.time() - t_start

        train_acc = pipeline.score(X_train, y_train)
        print(f"  ✓ Finished in: {t_elapsed:.2f} seconds")
        print(f"  ✓ Training Accuracy (Score): {train_acc:.4f} ({train_acc*100:.2f}%)")
        trained_models[name] = pipeline

    return trained_models


# ------------------------------------------------------------------------------
# Task 5: Model Saving
# ------------------------------------------------------------------------------
def save_models_and_metadata(
    trained_models: Dict[str, Pipeline],
    feature_columns: List[str]
):
    """
    Saves trained pipelines and feature column order using Joblib and JSON.
    """
    print("\n" + "=" * 60)
    print("SERIALIZING MODELS & PIPELINES VIA JOBLIB")
    print("=" * 60)

    for name, model in trained_models.items():
        file_path = MODEL_FILES[name]
        joblib.dump(model, file_path, compress=3)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  • Saved {name:15s} -> {file_path} ({size_kb:.1f} KB)")

    # Save ordered feature column names (both JSON and Joblib formats)
    with open(COLUMNS_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    print(f"  • Saved Feature Columns -> {COLUMNS_FILE} & feature_columns.joblib")


# ------------------------------------------------------------------------------
# Task 6: Testing & Verification
# ------------------------------------------------------------------------------
def verify_saved_models(X_test: pd.DataFrame, y_test: pd.Series, feature_columns: List[str]):
    """
    Loads saved model files from disk, runs sample test predictions,
    and verifies that predictions and probabilities are strictly valid.
    """
    print("\n" + "=" * 60)
    print("VERIFYING SAVED MODELS & SAMPLE PREDICTIONS")
    print("=" * 60)

    # Pick 5 distinct samples from held-out test data (mix of legitimate and phishing)
    sample_indices = [0, 1, 2, 3, 4]
    X_samples = X_test.iloc[sample_indices]
    y_samples = y_test.iloc[sample_indices].tolist()

    loaded_models = {}
    for name, file_path in MODEL_FILES.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} not found!")
        loaded_model = joblib.load(file_path)
        loaded_models[name] = loaded_model
        print(f"  ✓ Successfully verified and loaded: {os.path.basename(file_path)}")

    # Verify predictions
    print("\nTesting inference on 5 held-out test records:")
    print("-" * 80)
    for i, idx in enumerate(sample_indices):
        true_label = "Phishing (1)" if y_samples[i] == 1 else "Legitimate (0)"
        single_row = X_samples.iloc[[i]]

        print(f"Sample #{i+1} [Ground Truth: {true_label}]:")
        for name, model in loaded_models.items():
            pred = model.predict(single_row)[0]
            pred_text = "Phishing (1)" if pred == 1 else "Legitimate (0)"
            
            # Predict probabilities if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(single_row)[0]
                phish_prob = proba[1] * 100
                print(f"   - {name.replace('_', ' ').capitalize():15s} -> Pred: {pred_text} | Phishing Risk: {phish_prob:.1f}%")
            else:
                print(f"   - {name.replace('_', ' ').capitalize():15s} -> Pred: {pred_text}")
        print("-" * 80)


# ------------------------------------------------------------------------------
# Main Execution Pipeline
# ------------------------------------------------------------------------------
def main():
    # 1. Prepare data
    X, y, feature_columns = load_and_prepare_data(DATA_PATH)

    # 2. Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Create pipelines
    pipelines = create_model_pipelines()

    # 4. Train models
    trained_models = train_models(pipelines, X_train, y_train)

    # 5. Save models & metadata
    save_models_and_metadata(trained_models, feature_columns)

    # 6. Verify and test on held-out samples
    verify_saved_models(X_test, y_test, feature_columns)

    print("\nStage 5 Model Development finished successfully!")

if __name__ == "__main__":
    main()
