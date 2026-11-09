"""
Full End-to-End Execution Pipeline
==================================
Executes all project stages sequentially:
1. Dataset Generation (15,000 synthetic candidate records with injected historical bias)
2. Data Cleaning and Standardization (Casing, parsing experience, missingness, IQR outlier capping)
3. Feature Engineering and Preprocessing Pipeline
4. Exploratory Data Analysis and Historical Disparity Visualization
5. Model Training, Imbalance Strategy Comparison, Hyperparameter Tuning & Selection
6. Model Performance Evaluation (ROC, PR, Calibration, Confusion Matrix)
7. Algorithmic Fairness Auditing (Demographic Parity, Disparate Impact, Equal Opportunity, Mitigation)
8. Explainable AI (Global SHAP, Dependence, Local SHAP, LIME Concordance, Plain-English Narratives)
9. Candidate Scoring & Talent Ranking Demonstration
10. Automated Pytest Verification
"""

import sys
import os
import time

def print_header(title: str):
    print("\n" + "=" * 75)
    print(f"  >>> {title}")
    print("=" * 75 + "\n")

def run_pipeline():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print_header("STAGE 1: GENERATE DATASET (15,000 CANDIDATES)")
    from src.generate_dataset import main as gen_main
    gen_main()

    print_header("STAGE 2: DATA CLEANING & STANDARDIZATION")
    from src.cleaning import run_cleaning_pipeline
    run_cleaning_pipeline()

    print_header("STAGE 3: FEATURE ENGINEERING & PREPROCESSOR FIT")
    from src.features import main as feat_main
    feat_main()

    print_header("STAGE 4: EXPLORATORY DATA ANALYSIS (EDA)")
    from src.eda import run_eda_pipeline
    run_eda_pipeline()

    print_header("STAGE 5: MODEL TRAINING, HYPERPARAMETER TUNING & SELECTION")
    from src.train import run_training_pipeline
    run_training_pipeline()

    print_header("STAGE 6: PERFORMANCE EVALUATION")
    from src.evaluate import run_evaluation_pipeline
    run_evaluation_pipeline()

    print_header("STAGE 7: ALGORITHMIC FAIRNESS AUDIT & MITIGATION")
    from src.fairness import run_fairness_audit_pipeline
    run_fairness_audit_pipeline()

    print_header("STAGE 8: EXPLAINABLE AI (SHAP, LIME & RECRUITER NARRATIVES)")
    from src.explain import run_explainability_pipeline
    run_explainability_pipeline()

    print_header("STAGE 9: CANDIDATE SCORING & RANKING DEMO")
    from src.score import main as score_main
    score_main()

    print_header("STAGE 10: AUTOMATED TEST SUITE EXECUTION")
    import pytest
    test_ret = pytest.main(["-v", os.path.join(base_dir, "tests")])
    if test_ret != 0:
        print("[WARNING] Some tests did not pass completely. Please review logs.")
    else:
        print("[SUCCESS] All unit and integration tests passed cleanly!")

    elapsed = time.time() - start_time
    print("\n" + "#" * 75)
    print(f"  AI CANDIDATE SCORING PIPELINE COMPLETE in {elapsed:.1f}s")
    print("  Artifacts generated:")
    print("    - Models: models/best_model.joblib, preprocessor.joblib, metadata.json")
    print("    - Reports: reports/metrics.json, fairness_report.json, model_comparison.csv")
    print("    - Charts: reports/eda/, reports/fairness/, reports/explainability/")
    print("  Launch Web Application:")
    print("    streamlit run app/streamlit_app.py")
    print("#" * 75 + "\n")

if __name__ == "__main__":
    run_pipeline()

