"""
Candidate Scoring and Ranking Engine
====================================
Converts model prediction probabilities into a 0-100 suitability score,
assigns talent acquisition match tiers (Strong Match / Possible Match / Weak Match),
and ranks candidates per job role and overall.
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import joblib

def assign_tier(score: float) -> str:
    """Categorizes score into actionable recruiter match tiers."""
    if score >= 70.0:
        return "Strong Match"
    elif score >= 45.0:
        return "Possible Match"
    else:
        return "Weak Match"

def score_candidates(df: pd.DataFrame, model=None, preprocessor=None) -> pd.DataFrame:
    """
    Takes candidate DataFrame, transforms features using the fitted preprocessor,
    generates suitability scores (0-100), assigns match tiers, and computes per-role ranks.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    if model is None:
        model_path = os.path.join(models_dir, "best_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Train model first.")
        model = joblib.load(model_path)

    if preprocessor is None:
        prep_path = os.path.join(models_dir, "preprocessor.joblib")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Preprocessor not found at {prep_path}.")
        preprocessor = joblib.load(prep_path)

    df_scored = df.copy()

    # Transform features
    X_mat = preprocessor.transform(df_scored)
    
    # Predict probabilities of class 1 (shortlisted)
    probs = model.predict_proba(X_mat)[:, 1]

    # Convert to 0-100 suitability score
    scores = np.round(probs * 100.0, 1)
    df_scored['suitability_score'] = scores
    df_scored['predicted_probability'] = np.round(probs, 4)
    df_scored['match_tier'] = df_scored['suitability_score'].apply(assign_tier)

    # Rank candidates within their specific applied job role
    if 'job_role_applied' in df_scored.columns:
        df_scored['role_rank'] = df_scored.groupby('job_role_applied')['suitability_score'].rank(ascending=False, method='min').astype(int)
    else:
        df_scored['role_rank'] = 1

    # Overall pool rank
    df_scored['overall_rank'] = df_scored['suitability_score'].rank(ascending=False, method='min').astype(int)

    # Sort descending by score
    df_scored = df_scored.sort_values(by=['suitability_score', 'years_experience'], ascending=[False, False]).reset_index(drop=True)

    return df_scored

def score_single_candidate(candidate_dict: dict, model=None, preprocessor=None) -> dict:
    """Convenience helper to score a single candidate dictionary."""
    df_single = pd.DataFrame([candidate_dict])
    scored_df = score_candidates(df_single, model=model, preprocessor=preprocessor)
    return scored_df.iloc[0].to_dict()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(base_dir, "data", "processed", "cleaned_resumes.csv")
    if not os.path.exists(test_csv):
        print("Dataset not found. Please run the training pipeline first.")
        return

    df = pd.read_csv(test_csv).head(50)
    print(f"Scoring batch of {len(df)} candidates...")
    scored = score_candidates(df)

    cols_to_show = ['candidate_id', 'name', 'job_role_applied', 'suitability_score', 'match_tier', 'role_rank']
    avail = [c for c in cols_to_show if c in scored.columns]
    print(scored[avail].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
