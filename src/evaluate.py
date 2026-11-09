"""
Model Performance Evaluation Module
===================================
Computes comprehensive classification metrics and generates evaluation artifacts:
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score
- Confusion Matrix plot
- ROC Curve plot
- Precision-Recall Curve plot
- Probability Calibration Curve plot
- Exports detailed metrics to reports/metrics.json
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
import joblib

def evaluate_model_performance(model, X_test: np.ndarray, y_test: np.ndarray, output_dir: str = None) -> Dict[str, Any]:
    """Computes all classification metrics and plots evaluation curves."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1])
        }
    }

    # Save metrics.json
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_labels = np.array([
        [f"TN\n{cm[0, 0]:,d}", f"FP\n{cm[0, 1]:,d}"],
        [f"FN\n{cm[1, 0]:,d}", f"TP\n{cm[1, 1]:,d}"]
    ])
    sns.heatmap(cm, annot=cm_labels, fmt="", cmap="Blues", cbar=False, ax=ax, annot_kws={"fontsize": 12, "fontweight": "bold"})
    ax.set_title("Confusion Matrix (Holdout Test Set)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Ground Truth")
    ax.set_xticklabels(["Rejected (0)", "Shortlisted (1)"])
    ax.set_yticklabels(["Rejected (0)", "Shortlisted (1)"])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=200)
    plt.close(fig)

    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    baseline_rate = np.mean(y_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color="#10B981", lw=2.5, label=f"PR Curve (PR-AUC = {pr_auc:.3f})")
    ax.axhline(y=baseline_rate, color="#94A3B8", lw=1.5, linestyle="--", label=f"Baseline Rate ({baseline_rate:.2f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=200)
    plt.close(fig)

    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker='o', lw=2, color="#8B5CF6", label=f"Model (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Probability Calibration Curve (Reliability Diagram)", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=200)
    plt.close(fig)

    print("\n" + "=" * 50)
    print("Holdout Test Set Performance Metrics:")
    print("=" * 50)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  * {k.upper():<15}: {v:.4f}")
        else:
            print(f"  * Confusion Matrix: {v}")
    print(f"[SUCCESS] Metrics and evaluation curves exported to: {output_dir}")

    return metrics

def run_evaluation_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    proc_dir = os.path.join(base_dir, "data", "processed")

    model_path = os.path.join(models_dir, "best_model.joblib")
    X_test_path = os.path.join(proc_dir, "X_test.joblib")
    y_test_path = os.path.join(proc_dir, "y_test.joblib")

    if not os.path.exists(model_path) or not os.path.exists(X_test_path):
        from src.train import run_training_pipeline
        run_training_pipeline()

    print(f"Loading best model: {model_path}")
    model = joblib.load(model_path)
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    evaluate_model_performance(model, X_test, y_test)

if __name__ == "__main__":
    run_evaluation_pipeline()
