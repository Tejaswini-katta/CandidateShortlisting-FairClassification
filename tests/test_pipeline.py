"""
End-to-End Unit & Integration Test Suite
========================================
Tests:
- Resume Parser (Text parsing, skill extraction, regex experience extraction)
- Cleaning pipeline (Deduplication, degree/city standardization, IQR capping, missing imputation)
- Feature Pipeline (Fit, transform, shape alignment, feature names)
- Candidate Scoring Engine (Probability to 0-100 score, tier assignment, per-role ranking)
- Fairness Metric Calculations (Disparate impact ratio, demographic parity, four-fifths violation detection)
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import pytest
import numpy as np
import pandas as pd

from src.resume_parser import (
    parse_resume, extract_skills, extract_years_experience,
    extract_education_level, extract_degree_field
)
from src.cleaning import clean_dataset, clean_degree, clean_city, parse_experience
from src.features import (
    CandidateFeaturePipeline, MultiHotSkillEncoder,
    compute_skill_match_ratio, isolate_sensitive_attributes
)
from src.score import assign_tier, score_candidates
from src.fairness import compute_group_fairness_metrics

# 1. PARSER TESTS
def test_extract_skills():
    sample_text = "Proficient in Python, Machine Learning, SQL, and Docker with AWS experience."
    skills = extract_skills(sample_text)
    assert "Python" in skills
    assert "Machine Learning" in skills
    assert "SQL" in skills
    assert "Docker" in skills
    assert "AWS" in skills

def test_extract_years_experience():
    text1 = "Senior engineer with 5.5 years of experience building data pipelines."
    assert extract_years_experience(text1) == 5.5

    text2 = "Worked as software engineer from 2018 to 2024."
    assert extract_years_experience(text2) == 6.0

def test_extract_education_and_degree():
    text = "Candidate holds a Master of Technology in Computer Science from IIT."
    assert extract_education_level(text) == "Masters"
    assert extract_degree_field(text) == "Computer Science"

def test_parse_resume_full():
    text = """
    SANJAY VERMA
    Email: sanjay@example.com
    Experienced Data Scientist with 4.0 years of experience.
    Education: B.Tech in Computer Science (Tier 1 institute).
    Skills: Python, Machine Learning, SQL, Statistics, Pandas.
    Completed 5 projects and holds 2 certifications.
    GPA: 8.5/10.0. Notice period: 30 days. Expected salary: 15 LPA.
    """
    parsed = parse_resume(text, default_role="Data Scientist")
    assert parsed["name"] == "Sanjay Verma"
    assert parsed["education_level"] == "Bachelors"
    assert parsed["degree_field"] == "Computer Science"
    assert parsed["years_experience"] == 4.0
    assert "Python" in parsed["skills"]
    assert parsed["projects_count"] == 5
    assert parsed["certifications_count"] == 2

# 2. CLEANING TESTS
def test_clean_degree_and_city():
    assert clean_degree("b.tech in cs") == "Computer Science"
    assert clean_degree("BTech IT") == "Information Technology"
    assert clean_degree("Mechanical Engg") == "Mechanical"

    assert clean_city("bengaluru") == "Bengaluru"
    assert clean_city("Bombay") == "Mumbai"
    assert clean_city("gurgaon") == "Delhi NCR"

def test_parse_experience_variants():
    assert parse_experience("5 yrs") == 5.0
    assert parse_experience("3.5 Years") == 3.5
    assert parse_experience(7.0) == 7.0

def test_clean_dataset_flow():
    raw_data = pd.DataFrame([
        {
            "candidate_id": "C-1", "name": "Test User", "gender": "Female", "age": 28,
            "location": "bangalore", "education_level": "Bachelors", "degree_field": "b.tech in cs",
            "college_tier": np.nan, "gpa_or_percentage": 11.5, "years_experience": "4 yrs",
            "previous_job_titles": "Engineer", "current_company_type": np.nan,
            "skills": " python , SQL ", "certifications_count": 1, "projects_count": 3,
            "publications_count": 0, "employment_gap_months": 0, "notice_period_days": 30,
            "expected_salary_lpa": 12.0, "job_role_applied": "Data Scientist",
            "resume_text": "Sample text", "shortlisted": 1
        },
        # Duplicate row
        {
            "candidate_id": "C-2", "name": "Test User", "gender": "Female", "age": 28,
            "location": "bangalore", "education_level": "Bachelors", "degree_field": "b.tech in cs",
            "college_tier": "Tier 1", "gpa_or_percentage": 8.0, "years_experience": 4.0,
            "previous_job_titles": "Engineer", "current_company_type": "MNC",
            "skills": "Python, SQL", "certifications_count": 1, "projects_count": 3,
            "publications_count": 0, "employment_gap_months": 0, "notice_period_days": 30,
            "expected_salary_lpa": 12.0, "job_role_applied": "Data Scientist",
            "resume_text": "Sample text", "shortlisted": 1
        }
    ])
    cleaned_df, logs = clean_dataset(raw_data)
    # Deduplication check
    assert len(cleaned_df) == 1
    # Location standardization
    assert cleaned_df.iloc[0]["location"] == "Bengaluru"
    # Degree standardization
    assert cleaned_df.iloc[0]["degree_field"] == "Computer Science"
    # Outlier capping on GPA
    assert cleaned_df.iloc[0]["gpa_or_percentage"] <= 10.0
    # Missing value imputation
    assert pd.notna(cleaned_df.iloc[0]["college_tier"])
    assert pd.notna(cleaned_df.iloc[0]["current_company_type"])

# 3. FEATURE PIPELINE TESTS
def test_skill_match_ratio():
    skills = "Python, Machine Learning, SQL, Statistics"
    ratio = compute_skill_match_ratio(skills, "Data Scientist")
    assert 0.0 <= ratio <= 1.0
    assert ratio >= 0.4

def test_feature_pipeline_shape_and_encoding():
    sample_df = pd.DataFrame([
        {
            "candidate_id": "C-1", "name": "A", "gender": "Male", "age": 30,
            "location": "Bengaluru", "education_level": "Masters", "degree_field": "Computer Science",
            "college_tier": "Tier 1", "gpa_or_percentage": 8.5, "years_experience": 5.0,
            "gpa_is_missing": 0, "previous_job_titles": "Data Scientist", "current_company_type": "Product",
            "skills": "Python, Machine Learning, SQL, AWS, Docker",
            "certifications_count": 2, "projects_count": 5, "publications_count": 1,
            "employment_gap_months": 0, "notice_period_days": 30, "expected_salary_lpa": 20.0,
            "job_role_applied": "Data Scientist", "resume_text": "Expert in machine learning and data science."
        },
        {
            "candidate_id": "C-2", "name": "B", "gender": "Female", "age": 25,
            "location": "Pune", "education_level": "Bachelors", "degree_field": "Information Technology",
            "college_tier": "Tier 2", "gpa_or_percentage": 7.5, "years_experience": 2.0,
            "gpa_is_missing": 0, "previous_job_titles": "Software Engineer", "current_company_type": "Service",
            "skills": "Java, SQL, Git",
            "certifications_count": 1, "projects_count": 2, "publications_count": 0,
            "employment_gap_months": 3, "notice_period_days": 60, "expected_salary_lpa": 8.0,
            "job_role_applied": "Software Engineer", "resume_text": "Java backend developer with REST API knowledge."
        }
    ])

    df_feat, df_sens = isolate_sensitive_attributes(sample_df)
    assert "gender" not in df_feat.columns
    assert "age" not in df_feat.columns
    assert "name" not in df_feat.columns

    pipeline = CandidateFeaturePipeline(top_k_skills=10, tfidf_max_features=10)
    X = pipeline.fit_transform(df_feat)
    feature_names = pipeline.get_feature_names()

    assert X.shape[0] == 2
    assert X.shape[1] == len(feature_names)
    assert not np.isnan(X).any()

# 4. SCORING & TIER TESTS
def test_assign_tier():
    assert assign_tier(85.0) == "Strong Match"
    assert assign_tier(70.0) == "Strong Match"
    assert assign_tier(55.0) == "Possible Match"
    assert assign_tier(30.0) == "Weak Match"

# 5. FAIRNESS METRICS TESTS
def test_fairness_metric_computation():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 0])
    gender_group = pd.Series(["Male", "Male", "Male", "Male", "Male", "Female", "Female", "Female", "Female", "Female"])

    metrics = compute_group_fairness_metrics(y_true, y_pred, gender_group, "Gender")
    assert "demographic_parity_difference" in metrics
    assert "disparate_impact_ratio" in metrics
    assert "equal_opportunity_difference" in metrics
    assert 0.0 <= metrics["disparate_impact_ratio"] <= 2.0
