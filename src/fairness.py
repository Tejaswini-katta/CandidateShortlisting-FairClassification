"""
Fairness Auditing and Bias Mitigation Module
============================================
Performs algorithmic fairness auditing across protected attributes (Gender, Age Group, College Tier):
- Selection Rate
- Demographic Parity Difference and Ratio
- Disparate Impact Ratio (flags violations of the EEOC four-fifths / 80% rule)
- Equal Opportunity Difference (True Positive Rate gap)
- Equalized Odds Difference (max of TPR gap and FPR gap)
- False Positive Rate gap
Utilizes Fairlearn's MetricFrame and implements post-processing mitigation
(ThresholdOptimizer) with before-vs-after trade-off analysis.
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import (
    MetricFrame, selection_rate, demographic_parity_difference,
    demographic_parity_ratio, equalized_odds_difference, false_positive_rate,
    true_positive_rate
)
from fairlearn.postprocessing import ThresholdOptimizer
import joblib

def compute_group_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive_feature: pd.Series, group_name: str) -> Dict[str, Any]:
    """
    Computes standard algorithmic fairness metrics for a protected attribute:
    Selection rate, demographic parity, disparate impact ratio, equal opportunity, equalized odds.
    """
    # Filter out NaNs in sensitive feature
    valid_mask = ~sensitive_feature.isna()
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]
    sens = sensitive_feature[valid_mask]

    metric_frame = MetricFrame(
        metrics={
            'selection_rate': selection_rate,
            'accuracy': accuracy_score,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate
        },
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=sens
    )

    by_group = metric_frame.by_group.to_dict()

    # Demographic Parity Difference & Ratio (Disparate Impact)
    sel_rates = metric_frame.by_group['selection_rate']
    min_sel = float(sel_rates.min())
    max_sel = float(sel_rates.max())
    dp_diff = float(max_sel - min_sel)
    disparate_impact_ratio = float(min_sel / max_sel) if max_sel > 0 else 1.0

    # Equal Opportunity Difference (Max gap in TPR between groups)
    tpr_vals = metric_frame.by_group['true_positive_rate']
    eq_opp_diff = float(tpr_vals.max() - tpr_vals.min())

    # False Positive Rate Gap
    fpr_vals = metric_frame.by_group['false_positive_rate']
    fpr_diff = float(fpr_vals.max() - fpr_vals.min())

    # Equalized Odds Difference (Max of TPR gap and FPR gap)
    eq_odds_diff = max(eq_opp_diff, fpr_diff)

    # Four-fifths rule check
    violates_four_fifths = disparate_impact_ratio < 0.80

    return {
        "attribute": group_name,
        "demographic_parity_difference": round(dp_diff, 4),
        "disparate_impact_ratio": round(disparate_impact_ratio, 4),
        "four_fifths_rule_violated": bool(violates_four_fifths),
        "equal_opportunity_difference": round(eq_opp_diff, 4),
        "equalized_odds_difference": round(eq_odds_diff, 4),
        "false_positive_rate_gap": round(fpr_diff, 4),
        "group_metrics": {
            str(k): {
                "selection_rate": round(float(by_group['selection_rate'].get(k, 0)), 4),
                "accuracy": round(float(by_group['accuracy'].get(k, 0)), 4),
                "true_positive_rate": round(float(by_group['true_positive_rate'].get(k, 0)), 4),
                "false_positive_rate": round(float(by_group['false_positive_rate'].get(k, 0)), 4)
            }
            for k in metric_frame.by_group.index
        }
    }

def run_mitigation_experiment(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, sensitive_train: pd.Series, sensitive_test: pd.Series) -> Tuple[ThresholdOptimizer, pd.DataFrame]:
    """
    Applies Fairlearn ThresholdOptimizer to mitigate disparate impact and equal opportunity disparities
    across Gender while assessing accuracy and utility trade-offs.
    """
    print("\n--- Running Algorithmic Fairness Mitigation (ThresholdOptimizer: Equalized Odds) ---")
    
    # Baseline predictions (Unmitigated)
    y_pred_orig = model.predict(X_test)
    acc_orig = accuracy_score(y_test, y_pred_orig)
    f1_orig = f1_score(y_test, y_pred_orig, zero_division=0)
    
    mf_orig = compute_group_fairness_metrics(y_test, y_pred_orig, sensitive_test, "Gender")
    di_orig = mf_orig['disparate_impact_ratio']
    dp_orig = mf_orig['demographic_parity_difference']
    eq_orig = mf_orig['equal_opportunity_difference']

    # Post-processing mitigation
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        prefit=True,
        predict_method='predict_proba'
    )
    
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_pred_mitigated = optimizer.predict(X_test, sensitive_features=sensitive_test)

    acc_mit = accuracy_score(y_test, y_pred_mitigated)
    f1_mit = f1_score(y_test, y_pred_mitigated, zero_division=0)
    
    mf_mit = compute_group_fairness_metrics(y_test, y_pred_mitigated, sensitive_test, "Gender")
    di_mit = mf_mit['disparate_impact_ratio']
    dp_mit = mf_mit['demographic_parity_difference']
    eq_mit = mf_mit['equal_opportunity_difference']

    comparison_data = [
        {
            "Stage": "Before Mitigation (Baseline)",
            "Accuracy": round(float(acc_orig), 4),
            "F1_Score": round(float(f1_orig), 4),
            "Disparate_Impact_Ratio": round(float(di_orig), 4),
            "Demographic_Parity_Diff": round(float(dp_orig), 4),
            "Equal_Opportunity_Diff": round(float(eq_orig), 4),
            "Four_Fifths_Violated": "YES (Flagged)" if di_orig < 0.8 else "NO (Compliant)"
        },
        {
            "Stage": "After Mitigation (Equalized Odds)",
            "Accuracy": round(float(acc_mit), 4),
            "F1_Score": round(float(f1_mit), 4),
            "Disparate_Impact_Ratio": round(float(di_mit), 4),
            "Demographic_Parity_Diff": round(float(dp_mit), 4),
            "Equal_Opportunity_Diff": round(float(eq_mit), 4),
            "Four_Fifths_Violated": "YES (Flagged)" if di_mit < 0.8 else "NO (Compliant)"
        }
    ]
    df_mit = pd.DataFrame(comparison_data)
    print(df_mit.to_string(index=False))
    return optimizer, df_mit

def plot_fairness_visualizations(fairness_report: Dict[str, Any], df_mitigation: pd.DataFrame, output_dir: str):
    """Generates charts for group selection rates, disparate impact ratios, and mitigation trade-offs."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Selection Rates by Group
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    attributes = ["gender", "age_group", "college_tier"]
    titles = ["Gender Groups", "Age Groups", "College Tiers"]

    for i, (attr, title) in enumerate(zip(attributes, titles)):
        if attr in fairness_report:
            group_data = fairness_report[attr]["group_metrics"]
            labels = list(group_data.keys())
            rates = [group_data[k]["selection_rate"] * 100 for k in labels]
            
            bars = axes[i].bar(labels, rates, color="#3B82F6", width=0.5)
            for bar, rate in zip(bars, rates):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{rate:.1f}%", ha='center', va='bottom', fontweight='bold')
            axes[i].set_title(title, fontweight='bold')
            axes[i].set_ylabel("Selection Rate (%)" if i == 0 else "")
            axes[i].set_ylim(0, max(rates) * 1.25 if rates else 50)

    fig.suptitle("Model Selection Rates Across Demographic & Institutional Groups", fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "selection_rates_by_group.png"), dpi=200)
    plt.close(fig)

    # 2. Disparate Impact Ratios with 80% Four-Fifths Threshold
    di_data = {
        attr.replace("_", " ").title(): fairness_report[attr]["disparate_impact_ratio"]
        for attr in attributes if attr in fairness_report
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = list(di_data.keys())
    values = list(di_data.values())
    colors = ['#10B981' if v >= 0.80 else '#EF4444' for v in values]
    bars = ax.bar(categories, values, color=colors, width=0.45)
    ax.axhline(0.80, color='#DC2626', linestyle='--', lw=2, label="Four-Fifths Rule Threshold (0.80)")
    for bar, v in zip(bars, values):
        flag = "✓ Pass" if v >= 0.80 else "✗ Disparity"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{v:.2f}\n({flag})", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Disparate Impact Ratio (Min/Max Selection Rate)")
    ax.set_title("Disparate Impact Ratio by Sensitive Attribute", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "disparate_impact_ratios.png"), dpi=200)
    plt.close(fig)

    # 3. Before vs After Mitigation Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_to_plot = ["Accuracy", "F1_Score", "Disparate_Impact_Ratio", "Equal_Opportunity_Diff"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    before_vals = [df_mitigation.loc[0, col] for col in metrics_to_plot]
    after_vals = [df_mitigation.loc[1, col] for col in metrics_to_plot]

    ax.bar(x - width/2, before_vals, width, label='Before Mitigation', color='#94A3B8')
    ax.bar(x + width/2, after_vals, width, label='After Mitigation', color='#2563EB')
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "F1 Score", "Disparate Impact", "Eq Opportunity Gap"])
    ax.set_ylabel("Value")
    ax.set_title("Fairness vs Accuracy Trade-off (ThresholdOptimizer)", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fairness_mitigation_comparison.png"), dpi=200)
    plt.close(fig)

def run_fairness_audit_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    proc_dir = os.path.join(base_dir, "data", "processed")
    reports_dir = os.path.join(base_dir, "reports")
    fairness_dir = os.path.join(reports_dir, "fairness")

    os.makedirs(fairness_dir, exist_ok=True)

    # Load artifacts
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    X_test = joblib.load(os.path.join(proc_dir, "X_test.joblib"))
    y_test = joblib.load(os.path.join(proc_dir, "y_test.joblib"))
    X_train = joblib.load(os.path.join(proc_dir, "X_train.joblib"))
    y_train = joblib.load(os.path.join(proc_dir, "y_train.joblib"))

    sensitive_test = pd.read_csv(os.path.join(proc_dir, "sensitive_test.csv"))
    sensitive_train = pd.read_csv(os.path.join(proc_dir, "sensitive_train.csv"))

    # Prepare age brackets
    age_bins = [0, 29, 45, 120]
    age_labels = ["<30", "30-45", "45+"]
    sensitive_test['age_group'] = pd.cut(sensitive_test['age'], bins=age_bins, labels=age_labels)
    sensitive_train['age_group'] = pd.cut(sensitive_train['age'], bins=age_bins, labels=age_labels)

    y_pred = model.predict(X_test)

    # Compute fairness per sensitive attribute
    print("\nAuditing Fairness across Gender, Age Group, and College Tier...")
    gender_report = compute_group_fairness_metrics(y_test, y_pred, sensitive_test['gender'], "Gender")
    age_report = compute_group_fairness_metrics(y_test, y_pred, sensitive_test['age_group'], "Age Group")
    tier_report = compute_group_fairness_metrics(y_test, y_pred, sensitive_test['college_tier'], "College Tier")

    fairness_report = {
        "gender": gender_report,
        "age_group": age_report,
        "college_tier": tier_report
    }

    # Save fairness_report.json
    with open(os.path.join(reports_dir, "fairness_report.json"), "w", encoding="utf-8") as f:
        json.dump(fairness_report, f, indent=2)
    print(f"[SUCCESS] Fairness report saved to: {os.path.join(reports_dir, 'fairness_report.json')}")

    # Mitigation Experiment
    mitigated_model, df_mit = run_mitigation_experiment(
        model, X_train, y_train, X_test, y_test,
        sensitive_train['gender'], sensitive_test['gender']
    )
    df_mit.to_csv(os.path.join(reports_dir, "mitigation_comparison.csv"), index=False)
    joblib.dump(mitigated_model, os.path.join(models_dir, "mitigated_model.joblib"))

    # Plot Visualizations
    plot_fairness_visualizations(fairness_report, df_mit, fairness_dir)
    print(f"[SUCCESS] Fairness visualizations generated in: {fairness_dir}")

    return fairness_report, df_mit

if __name__ == "__main__":
    run_fairness_audit_pipeline()
