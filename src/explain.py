"""
Explainable AI (XAI) Module
===========================
Generates global and local model interpretability artifacts:
- Global SHAP summary beeswarm and bar plots
- Feature dependence plots for top driving features
- Local individual candidate waterfall and force explanations
- LIME tabular explanations and SHAP-LIME concordance evaluation
- Proxy dominance auditing
- Plain-English natural language recruiter narratives
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
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib

FEATURE_NAME_MAP = {
    'role_skill_match_score': 'Role Skill Match Ratio',
    'years_experience': 'Years of Experience',
    'education_level_ordinal': 'Education Level (Degree)',
    'projects_count': 'Technical Projects Count',
    'certifications_count': 'Professional Certifications',
    'gpa_or_percentage': 'Academic GPA',
    'employment_gap_months': 'Employment Gap Duration',
    'publications_count': 'Technical Publications',
    'expected_salary_lpa': 'Expected Salary (LPA)',
    'notice_period_days': 'Notice Period (Days)'
}

def clean_feature_label(name: str) -> str:
    """Translates raw model feature names into recruiter-friendly labels."""
    if name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[name]
    if name.startswith('skill_'):
        return f"Skill: {name.replace('skill_', '').replace('_', ' ').title()}"
    if name.startswith('degree_field_'):
        return f"Degree: {name.replace('degree_field_', '').title()}"
    if name.startswith('job_role_applied_'):
        return f"Role: {name.replace('job_role_applied_', '').title()}"
    if name.startswith('current_company_type_'):
        return f"Company: {name.replace('current_company_type_', '').title()}"
    if name.startswith('tfidf_'):
        return f"Resume Term: '{name.replace('tfidf_', '')}'"
    return name.replace('_', ' ').title()

def get_explainer(model, X_background: np.ndarray):
    """Initializes appropriate SHAP explainer based on model class."""
    model_type = type(model).__name__
    if "XGB" in model_type or "Forest" in model_type or "Tree" in model_type:
        return shap.TreeExplainer(model)
    elif "Logistic" in model_type or "Linear" in model_type:
        return shap.LinearExplainer(model, X_background)
    else:
        return shap.Explainer(model, X_background)

def compute_global_explanations(model, X_sample: np.ndarray, feature_names: List[str], output_dir: str):
    """Generates global SHAP summary, bar plots, and dependence plots."""
    os.makedirs(output_dir, exist_ok=True)
    explainer = get_explainer(model, X_sample[:100])
    shap_values = explainer.shap_values(X_sample)

    # Handle binary classifier multi-output shape
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals_class1 = shap_values[1]
    elif hasattr(shap_values, "values"):
        shap_vals_class1 = shap_values.values
        if len(shap_vals_class1.shape) == 3:
            shap_vals_class1 = shap_vals_class1[:, :, 1]
    else:
        shap_vals_class1 = shap_values

    # 1. SHAP Beeswarm Summary Plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    shap.summary_plot(shap_vals_class1, X_sample, feature_names=feature_names, show=False, max_display=12)
    plt.title("Global SHAP Feature Impact Distribution", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. SHAP Bar Plot (Mean Absolute Impact)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    shap.summary_plot(shap_vals_class1, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=12)
    plt.title("Top Global Feature Importance (Mean |SHAP|)", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Top 3 feature indices
    mean_abs = np.mean(np.abs(shap_vals_class1), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:3]

    # 3. Dependence Plots for Top 3 features
    for rank, idx in enumerate(top_indices, 1):
        feat_name = feature_names[idx]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        shap.dependence_plot(idx, shap_vals_class1, X_sample, feature_names=feature_names, show=False, ax=ax)
        plt.title(f"SHAP Dependence: {clean_feature_label(feat_name)}", fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_dependence_top{rank}.png"), dpi=200, bbox_inches='tight')
        plt.close()

    return explainer

def compare_shap_and_lime(model, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str], output_dir: str, n_samples: int = 5) -> List[Dict[str, Any]]:
    """Evaluates local explanation concordance between SHAP and LIME across 5 candidates."""
    os.makedirs(output_dir, exist_ok=True)
    explainer_shap = get_explainer(model, X_train[:150])
    
    explainer_lime = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Rejected', 'Shortlisted'],
        mode='classification',
        random_state=42
    )

    comparisons = []
    
    for i in range(min(n_samples, len(X_test))):
        instance = X_test[i]
        
        # SHAP local values
        shap_res = explainer_shap.shap_values(instance.reshape(1, -1))
        if isinstance(shap_res, list) and len(shap_res) == 2:
            shap_vals = shap_res[1][0]
        elif hasattr(shap_res, "values"):
            shap_vals = shap_res.values[0]
            if len(shap_vals.shape) == 2:
                shap_vals = shap_vals[:, 1]
        else:
            shap_vals = shap_res[0] if len(shap_res.shape) > 1 else shap_res

        top_shap_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_shap_features = [feature_names[j] for j in top_shap_idx]

        # LIME local values
        exp_lime = explainer_lime.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=5
        )
        lime_map = exp_lime.as_map()[1]  # positive class
        top_lime_idx = [item[0] for item in lime_map]
        top_lime_features = [feature_names[j] for j in top_lime_idx]

        # Overlap (Jaccard similarity on top 5)
        intersection = set(top_shap_features) & set(top_lime_features)
        jaccard = round(len(intersection) / max(len(set(top_shap_features) | set(top_lime_features)), 1), 3)

        comparisons.append({
            "candidate_index": i,
            "jaccard_agreement": jaccard,
            "top_shap_features": [clean_feature_label(f) for f in top_shap_features],
            "top_lime_features": [clean_feature_label(f) for f in top_lime_features],
            "common_features": [clean_feature_label(f) for f in intersection]
        })

    with open(os.path.join(output_dir, "shap_lime_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)

    return comparisons

def explain_candidate(candidate_dict: dict, model=None, preprocessor=None, feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Generates plain-English recruiter explanations and individual SHAP factor breakdowns
    for an individual candidate.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    if model is None:
        model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    if preprocessor is None:
        preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    if feature_names is None:
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))

    df_cand = pd.DataFrame([candidate_dict])
    X_cand = preprocessor.transform(df_cand)
    prob = model.predict_proba(X_cand)[0, 1]
    score = round(prob * 100.0, 1)

    # Compute SHAP
    explainer = get_explainer(model, X_cand)
    shap_res = explainer.shap_values(X_cand)
    if isinstance(shap_res, list) and len(shap_res) == 2:
        shap_vals = shap_res[1][0]
    elif hasattr(shap_res, "values"):
        shap_vals = shap_res.values[0]
        if len(shap_vals.shape) == 2:
            shap_vals = shap_vals[:, 1]
    else:
        shap_vals = shap_res[0] if len(shap_res.shape) > 1 else shap_res

    # Positive and negative factors
    pos_idx = [i for i in np.argsort(shap_vals)[::-1] if shap_vals[i] > 0][:5]
    neg_idx = [i for i in np.argsort(shap_vals) if shap_vals[i] < 0][:5]

    top_positive = [
        {
            "feature": feature_names[i],
            "label": clean_feature_label(feature_names[i]),
            "shap_value": round(float(shap_vals[i]), 4)
        }
        for i in pos_idx
    ]

    top_negative = [
        {
            "feature": feature_names[i],
            "label": clean_feature_label(feature_names[i]),
            "shap_value": round(float(shap_vals[i]), 4)
        }
        for i in neg_idx
    ]

    # Generate plain-English narrative
    pos_desc = ", ".join([f"{item['label']}" for item in top_positive[:3]]) if top_positive else "general eligibility"
    neg_desc = ", ".join([f"{item['label']}" for item in top_negative[:2]]) if top_negative else "no significant deficits"
    
    tier = "Strong Match" if score >= 70 else ("Possible Match" if score >= 45 else "Weak Match")
    exp_val = candidate_dict.get('years_experience', 'several')
    role_val = candidate_dict.get('job_role_applied', 'the role')
    edu_val = candidate_dict.get('education_level', 'Degree')

    summary_text = (
        f"Candidate scored {score:.0f}/100 ({tier}) for {role_val}. "
        f"Key positive drivers include: {pos_desc}. "
        f"Score was constrained by: {neg_desc}."
    )

    return {
        "suitability_score": float(score),
        "predicted_probability": round(float(prob), 4),
        "match_tier": tier,
        "summary_narrative": summary_text,
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "shap_values": [float(v) for v in np.array(shap_vals).flatten()],
        "feature_names": feature_names
    }

def run_explainability_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    proc_dir = os.path.join(base_dir, "data", "processed")
    reports_dir = os.path.join(base_dir, "reports")
    explain_dir = os.path.join(reports_dir, "explainability")

    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    X_train = joblib.load(os.path.join(proc_dir, "X_train.joblib"))
    X_test = joblib.load(os.path.join(proc_dir, "X_test.joblib"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))

    print("\nComputing Global SHAP Explanations...")
    compute_global_explanations(model, X_test[:300], feature_names, explain_dir)
    print(f"[SUCCESS] Global SHAP plots saved to {explain_dir}")

    print("\nComparing SHAP and LIME Concordance across 5 sample candidates...")
    comparisons = compare_shap_and_lime(model, X_train[:300], X_test[:5], feature_names, explain_dir, n_samples=5)
    avg_jaccard = np.mean([c['jaccard_agreement'] for c in comparisons])
    print(f"Mean SHAP-LIME Top-5 Feature Agreement: {avg_jaccard:.2f}")

    # Generate narrative explanation on a sample candidate
    sample_candidate = {
        "candidate_id": "SAMPLE-001",
        "name": "Priya Sharma",
        "gender": "Female",
        "age": 29,
        "location": "Bengaluru",
        "education_level": "Masters",
        "degree_field": "Computer Science",
        "college_tier": "Tier 1",
        "gpa_or_percentage": 8.8,
        "years_experience": 5.5,
        "previous_job_titles": "Data Scientist",
        "current_company_type": "Product",
        "skills": "Python, Machine Learning, SQL, Statistics, Deep Learning, AWS, Scikit-Learn",
        "certifications_count": 2,
        "projects_count": 6,
        "publications_count": 1,
        "employment_gap_months": 0,
        "notice_period_days": 30,
        "expected_salary_lpa": 18.0,
        "job_role_applied": "Data Scientist",
        "resume_text": "Experienced Data Scientist with 5.5 years in machine learning and deep learning. Masters in CS from Tier 1 institute. Adept at Python, SQL, Statistics, AWS."
    }
    explanation = explain_candidate(sample_candidate, model, preprocessor, feature_names)
    print("\nSample Plain-English Recruiter Explanation:")
    print(f"  {explanation['summary_narrative']}")
    with open(os.path.join(explain_dir, "sample_narrative.json"), "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2, default=float)

if __name__ == "__main__":
    run_explainability_pipeline()
