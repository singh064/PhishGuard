"""
PhishGuard – Model Evaluation and Comparison Module
====================================================
Evaluates the three trained classification models (Logistic Regression, SVM,
Decision Tree) on the identical held-out test dataset (20% stratified split).

Calculates:
- Accuracy, Precision, Recall (Phishing pos_label=1), F1-Score, ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)
- Detailed Classification Reports

Generates and saves:
- Accuracy comparison bar chart
- Precision, Recall, F1-Score multi-bar chart
- Individual & combined confusion matrix heatmaps
- Results summary CSV
"""

import os
import sys
import json
import time
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------------------------
# Paths Configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "features_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_model.pkl"),
    "Support Vector Machine": os.path.join(MODELS_DIR, "svm_model.pkl"),
    "Decision Tree": os.path.join(MODELS_DIR, "decision_tree_model.pkl")
}

# ------------------------------------------------------------------------------
# 1. Data Ingestion & Test Set Reconstruction
# ------------------------------------------------------------------------------
def load_test_dataset():
    print(f"Loading feature dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    y_raw = df["Label"]
    X = df.drop(columns=["Label"])
    if "URL" in X.columns:
        X = X.drop(columns=["URL"])
        
    # Target label mapping verification
    # 0 = Legitimate ('good'), 1 = Phishing ('bad')
    label_mapping = {"good": 0, "bad": 1}
    y = y_raw.map(label_mapping)
    
    print(f"Target Label Mapping Verified: 0 = Legitimate, 1 = Phishing (Positive Class)")
    
    # Exact stratified split matching train_model.py
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Held-Out Test Set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    print(f"  • Legitimate (0): {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"  • Phishing (1):   {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.2f}%)")
    
    return X_test, y_test

# ------------------------------------------------------------------------------
# 2. Model Evaluation
# ------------------------------------------------------------------------------
def evaluate_all_models(X_test: pd.DataFrame, y_test: pd.Series):
    results = {}
    reports = {}
    cms = {}
    
    print("\n" + "=" * 70)
    print("EVALUATING TRAINED MODELS ON HELD-OUT TEST DATA")
    print("=" * 70)
    
    for name, path in MODEL_FILES.items():
        print(f"\nLoading {name} from {os.path.basename(path)}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found!")
            
        model = joblib.load(path)
        
        # Predictions
        t0 = time.time()
        y_pred = model.predict(X_test)
        infer_time = time.time() - t0
        
        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = np.nan
            
        # Core Classification Metrics (pos_label=1 for Phishing)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) # True Negative Rate / Legitimate Recall
        
        results[name] = {
            "Accuracy": acc,
            "Precision (Phishing)": prec,
            "Recall (Phishing)": rec,
            "F1-Score (Phishing)": f1,
            "Specificity (Legitimate)": spec,
            "ROC-AUC": auc,
            "True Negatives (TN)": int(tn),
            "False Positives (FP)": int(fp),
            "False Negatives (FN)": int(fn),
            "True Positives (TP)": int(tp),
            "Inference Time (s)": infer_time
        }
        cms[name] = cm
        reports[name] = classification_report(
            y_test, y_pred,
            target_names=["Legitimate (0)", "Phishing (1)"],
            digits=4
        )
        
        print(f"--- {name} Results ---")
        print(f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision (Phish):   {prec:.4f} ({prec*100:.2f}%)")
        print(f"Recall (Phish):      {rec:.4f} ({rec*100:.2f}%)")
        print(f"F1-Score (Phish):    {f1:.4f} ({f1*100:.2f}%)")
        print(f"ROC-AUC:             {auc:.4f}")
        print(f"Confusion Matrix:    TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
        print(f"False Negatives:     {fn:,} (Uncaught Phishing Sites)")
        print(f"Classification Report:\n{reports[name]}")

    df_results = pd.DataFrame(results).T
    return df_results, cms, reports

# ------------------------------------------------------------------------------
# 3. Visualizations
# ------------------------------------------------------------------------------
def generate_visualizations(df_results: pd.DataFrame, cms: Dict[str, np.ndarray]):
    print("\n" + "=" * 70)
    print("GENERATING EVALUATION CHARTS & HEATMAPS")
    print("=" * 70)
    
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = ["#3498db", "#9b59b6", "#2ecc71"]
    models = list(df_results.index)
    
    # 1. Model Accuracy Comparison Bar Chart
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=df_results.index,
        y=df_results["Accuracy"],
        palette=palette
    )
    plt.title("Model Accuracy Comparison (Held-Out Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.ylim(0.75, 0.95)
    for p in ax.patches:
        val = p.get_height()
        ax.annotate(f"{val*100:.2f}%",
                    (p.get_x() + p.get_width() / 2., val),
                    ha="center", va="bottom", fontsize=11, fontweight="bold", xytext=(0, 4),
                    textcoords="offset points")
    plt.tight_layout()
    acc_path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved: {acc_path}")
    
    # 2. Precision, Recall, and F1-Score Comparison Chart
    metric_cols = ["Precision (Phishing)", "Recall (Phishing)", "F1-Score (Phishing)"]
    plot_df = df_results[metric_cols].reset_index().melt(
        id_vars="index", var_name="Metric", value_name="Score"
    )
    plot_df["Metric"] = plot_df["Metric"].str.replace(" (Phishing)", "", regex=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=plot_df,
        x="index",
        y="Score",
        hue="Metric",
        palette=["#e74c3c", "#f39c12", "#2980b9"]
    )
    plt.title("Precision, Recall & F1-Score Comparison (Phishing Detection)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.legend(title="Metric", fontsize=11, title_fontsize=11, loc="lower right")
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val*100:.1f}%",
                        (p.get_x() + p.get_width() / 2., val),
                        ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 3),
                        textcoords="offset points")
    plt.tight_layout()
    metrics_path = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"Saved: {metrics_path}")
    
    # 3. Individual Confusion Matrix Heatmaps
    file_slugs = {
        "Logistic Regression": "confusion_matrix_logistic.png",
        "Support Vector Machine": "confusion_matrix_svm.png",
        "Decision Tree": "confusion_matrix_decision_tree.png"
    }
    
    for name, cm in cms.items():
        plt.figure(figsize=(6, 5))
        # Formatted labels showing counts and percentages
        cm_sum = np.sum(cm)
        labels = [
            [f"TN\n{cm[0,0]:,}\n({cm[0,0]/cm_sum*100:.1f}%)", f"FP\n{cm[0,1]:,}\n({cm[0,1]/cm_sum*100:.1f}%)"],
            [f"FN\n{cm[1,0]:,}\n({cm[1,0]/cm_sum*100:.1f}%)", f"TP\n{cm[1,1]:,}\n({cm[1,1]/cm_sum*100:.1f}%)"]
        ]
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=True,
            xticklabels=["Pred: Legitimate", "Pred: Phishing"],
            yticklabels=["True: Legitimate", "True: Phishing"],
            annot_kws={"fontsize": 11, "weight": "bold"}
        )
        plt.title(f"Confusion Matrix: {name}", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("Ground Truth Class", fontsize=11)
        plt.tight_layout()
        cm_path = os.path.join(RESULTS_DIR, file_slugs[name])
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved: {cm_path}")
        
    # 4. Combined Confusion Matrix Panel (Side-by-Side Comparison)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm_sum = np.sum(list(cms.values())[0])
    
    for idx, (name, cm) in enumerate(cms.items()):
        labels = [
            [f"TN\n{cm[0,0]:,}\n({cm[0,0]/cm_sum*100:.1f}%)", f"FP\n{cm[0,1]:,}\n({cm[0,1]/cm_sum*100:.1f}%)"],
            [f"FN\n{cm[1,0]:,}\n({cm[1,0]/cm_sum*100:.1f}%)", f"TP\n{cm[1,1]:,}\n({cm[1,1]/cm_sum*100:.1f}%)"]
        ]
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=(idx == 2), ax=axes[idx],
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"],
            annot_kws={"fontsize": 10, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}", fontsize=13, fontweight="bold", pad=10)
        axes[idx].set_xlabel("Predicted Class", fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel("Actual Class", fontsize=11)
        else:
            axes[idx].set_ylabel("")
            
    fig.suptitle("PhishGuard Confusion Matrices Comparison", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    comb_path = os.path.join(RESULTS_DIR, "confusion_matrices_combined.png")
    plt.savefig(comb_path, dpi=300)
    plt.close()
    print(f"Saved: {comb_path}")

# ------------------------------------------------------------------------------
# 4. Save Results to CSV & Markdown Summary
# ------------------------------------------------------------------------------
def save_results(df_results: pd.DataFrame):
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_results.to_csv(csv_path, index=True, index_label="Model")
    print(f"\nSaved evaluation metrics table to: {csv_path}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    X_test, y_test = load_test_dataset()
    df_results, cms, reports = evaluate_all_models(X_test, y_test)
    generate_visualizations(df_results, cms)
    save_results(df_results)
    
    print("\n" + "=" * 70)
    print("STAGE 6 EVALUATION SUMMARY TABLE")
    print("=" * 70)
    print(df_results[["Accuracy", "Precision (Phishing)", "Recall (Phishing)", "F1-Score (Phishing)", "ROC-AUC", "False Negatives (FN)"]])
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
