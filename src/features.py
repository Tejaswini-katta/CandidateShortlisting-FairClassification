"""
Feature Engineering and Pipeline Module
=======================================
Constructs scikit-learn compatible preprocessors for candidate feature encoding:
- Ordinal encoding for education level
- One-hot encoding for degree fields, applied role, and company type
- Multi-hot encoding for top 40 skills and explicit role-skill match computation
- TF-IDF vectorization for resume text (max 60 components)
- Robust numeric scaling
- Sensitive attribute isolation (gender, age, candidate_id, name kept strictly out of training)
- Proxy variable detection and diagnostic auditing
"""

import os
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

ROLE_REQUIRED_SKILLS: Dict[str, List[str]] = {
    "Data Scientist": ["Python", "Machine Learning", "SQL", "Statistics", "Deep Learning", "Pandas", "Scikit-Learn", "NLP", "Data Visualization", "AWS"],
    "ML Engineer": ["Python", "Machine Learning", "Deep Learning", "Docker", "AWS", "MLOps", "PyTorch", "Kubernetes", "Git", "SQL"],
    "Data Analyst": ["SQL", "Excel", "Power BI", "Python", "Tableau", "Data Visualization", "Statistics", "Communication", "Reporting", "Pandas"],
    "Software Engineer": ["Java", "Python", "Docker", "AWS", "Git", "REST APIs", "SQL", "Data Structures", "System Design", "CI/CD"],
    "Business Analyst": ["Excel", "SQL", "Power BI", "Agile", "Communication", "Requirements Gathering", "Tableau", "Problem Solving", "Jira", "Process Mapping"]
}

EDUCATION_ORDER: List[str] = ["High School", "Diploma", "Bachelors", "Masters", "PhD"]

class MultiHotSkillEncoder(BaseEstimator, TransformerMixin):
    """Encodes comma-separated skill strings into binary flags for top K skills."""
    def __init__(self, top_k: int = 40):
        self.top_k = top_k
        self.top_skills_: List[str] = []

    def fit(self, X, y=None):
        skill_counts: Dict[str, int] = {}
        for row in X:
            if pd.isna(row) or not str(row).strip():
                continue
            skills = [s.strip().title() for s in str(row).split(",") if s.strip()]
            for s in set(skills):
                skill_counts[s] = skill_counts.get(s, 0) + 1
        
        # Sort by frequency
        sorted_skills = sorted(skill_counts.items(), key=lambda item: item[1], reverse=True)
        self.top_skills_ = [k for k, v in sorted_skills[:self.top_k]]
        return self

    def transform(self, X):
        matrix = np.zeros((len(X), len(self.top_skills_)), dtype=np.float32)
        for i, row in enumerate(X):
            if pd.isna(row) or not str(row).strip():
                continue
            candidate_skills = {s.strip().title() for s in str(row).split(",") if s.strip()}
            for j, skill in enumerate(self.top_skills_):
                if skill in candidate_skills:
                    matrix[i, j] = 1.0
        return matrix

    def get_feature_names_out(self, input_features=None):
        return [f"skill_{s.lower().replace(' ', '_').replace('/', '_')}" for s in self.top_skills_]

def compute_skill_match_ratio(skills_str: any, role: str) -> float:
    """Computes fraction of target role requirements satisfied by candidate skills."""
    if pd.isna(skills_str) or not str(skills_str).strip():
        return 0.1
    candidate_skills = {s.strip().title() for s in str(skills_str).split(",") if s.strip()}
    reqs = ROLE_REQUIRED_SKILLS.get(role, ROLE_REQUIRED_SKILLS["Data Scientist"])
    match_count = sum(1 for req in reqs if req.title() in candidate_skills)
    return round(float(match_count / len(reqs)), 3)

class CandidateFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end transformer that parses structured candidate fields and produces
    the standardized model feature matrix while strictly excluding sensitive attributes.
    """
    def __init__(self, top_k_skills: int = 40, tfidf_max_features: int = 60):
        self.top_k_skills = top_k_skills
        self.tfidf_max_features = tfidf_max_features
        
        self.scaler = StandardScaler()
        self.skill_encoder = MultiHotSkillEncoder(top_k=top_k_skills)
        self.cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.edu_encoder = OrdinalEncoder(categories=[EDUCATION_ORDER], handle_unknown='use_encoded_value', unknown_value=-1)
        self.tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english', ngram_range=(1, 2))
        
        self.num_cols = [
            'years_experience', 'gpa_or_percentage', 'gpa_is_missing',
            'certifications_count', 'projects_count', 'publications_count',
            'employment_gap_months', 'notice_period_days', 'expected_salary_lpa',
            'role_skill_match_score'
        ]
        self.cat_cols = ['degree_field', 'job_role_applied', 'current_company_type']
        self.feature_names_: List[str] = []

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df_mod = df.copy()
        if 'role_skill_match_score' not in df_mod.columns:
            match_scores = [
                compute_skill_match_ratio(row.get('skills', ''), row.get('job_role_applied', 'Data Scientist'))
                for _, row in df_mod.iterrows()
            ]
            df_mod['role_skill_match_score'] = match_scores
        
        # Ensure all required numeric columns exist
        for col in self.num_cols:
            if col not in df_mod.columns:
                df_mod[col] = 0.0
                
        # Default missing values safely
        df_mod['education_level'] = df_mod['education_level'].fillna("Bachelors")
        df_mod['resume_text'] = df_mod['resume_text'].fillna("")
        return df_mod

    def fit(self, X: pd.DataFrame, y=None):
        df = self._prepare_df(X)
        
        # Fit scaler on numeric features
        self.scaler.fit(df[self.num_cols].values)
        
        # Fit ordinal encoder on education
        self.edu_encoder.fit(df[['education_level']])
        
        # Fit one-hot encoder on nominal categorical features
        self.cat_encoder.fit(df[self.cat_cols])
        
        # Fit multi-hot skills encoder
        self.skill_encoder.fit(df['skills'].values)
        
        # Fit TF-IDF on resume text
        self.tfidf.fit(df['resume_text'].values)
        
        # Compile feature names
        num_names = self.num_cols
        edu_names = ['education_level_ordinal']
        cat_names = list(self.cat_encoder.get_feature_names_out(self.cat_cols))
        skill_names = list(self.skill_encoder.get_feature_names_out())
        tfidf_names = [f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]
        
        self.feature_names_ = num_names + edu_names + cat_names + skill_names + tfidf_names
        return self

    def transform(self, X: pd.DataFrame):
        df = self._prepare_df(X)
        
        X_num = self.scaler.transform(df[self.num_cols].values)
        X_edu = self.edu_encoder.transform(df[['education_level']])
        X_cat = self.cat_encoder.transform(df[self.cat_cols])
        X_skills = self.skill_encoder.transform(df['skills'].values)
        X_tfidf = self.tfidf.transform(df['resume_text'].values).toarray()
        
        return np.hstack([X_num, X_edu, X_cat, X_skills, X_tfidf])

    def get_feature_names(self) -> List[str]:
        return self.feature_names_

def isolate_sensitive_attributes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates the candidate dataset into:
    1. Features DataFrame (no gender, age, name, candidate_id)
    2. Sensitive Attributes DataFrame (for fairness auditing and proxy analysis)
    """
    sensitive_cols = ['candidate_id', 'name', 'gender', 'age', 'college_tier', 'location']
    present_sensitive = [c for c in sensitive_cols if c in df.columns]
    df_sensitive = df[present_sensitive].copy()
    
    # Feature columns (exclude sensitive attributes and target)
    exclude = set(sensitive_cols + ['shortlisted'])
    feature_cols = [c for c in df.columns if c not in exclude]
    df_features = df[feature_cols].copy()
    
    return df_features, df_sensitive

def audit_proxy_variables(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits correlation and associations between sensitive attributes
    (gender, age, college_tier, location) and model features to identify potential proxy leakage.
    """
    report = {}
    
    # Numeric correlations with age
    num_cols = ['years_experience', 'gpa_or_percentage', 'employment_gap_months', 'expected_salary_lpa']
    age_corrs = {}
    for col in num_cols:
        if col in df.columns and 'age' in df.columns:
            val = float(df[col].corr(df['age']))
            age_corrs[col] = round(val, 4)
    report['age_correlations'] = age_corrs

    # Gender breakdown across degree fields
    if 'gender' in df.columns and 'degree_field' in df.columns:
        gender_degree_ct = pd.crosstab(df['gender'], df['degree_field'], normalize='index') * 100
        report['gender_degree_distribution_pct'] = gender_degree_ct.round(2).to_dict()

    # College tier vs expected salary & projects
    if 'college_tier' in df.columns:
        tier_stats = df.groupby('college_tier')[['expected_salary_lpa', 'projects_count']].mean().round(2).to_dict()
        report['college_tier_stats'] = tier_stats

    return report

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_csv = os.path.join(base_dir, "data", "processed", "cleaned_resumes.csv")
    
    if not os.path.exists(clean_csv):
        print(f"Cleaned dataset not found at {clean_csv}. Running cleaning first...")
        from src.cleaning import run_cleaning_pipeline
        run_cleaning_pipeline()

    print(f"Loading cleaned dataset: {clean_csv}")
    df = pd.read_csv(clean_csv)

    df_features, df_sensitive = isolate_sensitive_attributes(df)
    print(f"Isolated Features shape: {df_features.shape}")
    print(f"Isolated Sensitive Attributes shape: {df_sensitive.shape}")

    # Build and fit pipeline
    print("Fitting candidate feature pipeline...")
    pipeline = CandidateFeaturePipeline(top_k_skills=40, tfidf_max_features=60)
    X = pipeline.fit_transform(df_features)
    feature_names = pipeline.get_feature_names()
    print(f"Encoded Feature Matrix shape: {X.shape}")
    print(f"Total features extracted: {len(feature_names)}")

    # Audit proxy relationships
    proxy_report = audit_proxy_variables(df)
    print("\nProxy Variable Audit Findings:")
    print(f"  * Age vs Experience correlation: {proxy_report['age_correlations'].get('years_experience', 'N/A')}")
    print(f"  * Age vs Expected Salary correlation: {proxy_report['age_correlations'].get('expected_salary_lpa', 'N/A')}")

    # Save artifacts
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.joblib"))
    joblib.dump(pipeline.skill_encoder.top_skills_, os.path.join(models_dir, "top_skills.joblib"))
    print(f"[SUCCESS] Preprocessor and feature names saved to {models_dir}")

if __name__ == "__main__":
    main()

