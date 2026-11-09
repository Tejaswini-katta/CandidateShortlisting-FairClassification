"""
Model Training and Hyperparameter Tuning Module
================================================
- Stratified 80/20 train/test split (random_state=42)
- Sensitive attributes preservation aligned with test set
- 5-fold Stratified Cross-Validation on training set
- Imbalance strategy comparison (none vs class_weight='balanced' vs SMOTE on training set only)
- Logistic Regression, Random Forest, and XGBoost with light hyperparameter tuning (RandomizedSearchCV)
- Selection and export of best model, preprocessor, and comprehensive training metadata
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

from src.features import CandidateFeaturePipeline, isolate_sensitive_attributes

def compare_imbalance_strategies(X_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    """
    Compares class imbalance strategies using 5-fold Stratified CV on the training data only:
    1. No handling (standard weights)
    2. Balanced class weights
    3. SMOTE applied strictly within training folds (never before split)
    """
    print("\n--- Comparing Class Imbalance Strategies (5-Fold CV on Training Set) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    strategies = ["None (Unweighted)", "class_weight='balanced'", "SMOTE (Training Fold Only)"]
    records = []

    for strat in strategies:
        f1_scores = []
        recall_scores = []
        roc_aucs = []
        precision_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            if strat == "None (Unweighted)":
                clf = LogisticRegression(max_iter=500, random_state=42)
                clf.fit(X_tr, y_tr)
            elif strat == "class_weight='balanced'":
                clf = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
                clf.fit(X_tr, y_tr)
            elif strat == "SMOTE (Training Fold Only)":
                smote = SMOTE(random_state=42)
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
                clf = LogisticRegression(max_iter=500, random_state=42)
                clf.fit(X_tr_res, y_tr_res)

            preds = clf.predict(X_val)
            probs = clf.predict_proba(X_val)[:, 1]

            f1_scores.append(f1_score(y_val, preds, zero_division=0))
            recall_scores.append(recall_score(y_val, preds, zero_division=0))
            precision_scores.append(precision_score(y_val, preds, zero_division=0))
            roc_aucs.append(roc_auc_score(y_val, probs))

        records.append({
            "Strategy": strat,
            "Mean_CV_F1": round(float(np.mean(f1_scores)), 4),
            "Mean_CV_Recall": round(float(np.mean(recall_scores)), 4),
            "Mean_CV_Precision": round(float(np.mean(precision_scores)), 4),
            "Mean_CV_ROC_AUC": round(float(np.mean(roc_aucs)), 4)
        })

    df_strat = pd.DataFrame(records)
    print(df_strat.to_string(index=False))
    return df_strat

def tune_and_compare_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Tunes Logistic Regression, Random Forest, and XGBoost using RandomizedSearchCV with 5-fold Stratified CV,
    evaluates on the unseen test set, and generates a comparative benchmark.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    # 1. Logistic Regression
    lr_param_dist = {
        'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    lr_base = LogisticRegression(max_iter=800, class_weight='balanced', random_state=42)
    lr_search = RandomizedSearchCV(lr_base, lr_param_dist, n_iter=6, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)

    # 2. Random Forest
    rf_param_dist = {
        'n_estimators': [80, 120, 150],
        'max_depth': [8, 12, 16, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=6, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)

    # 3. XGBoost
    xgb_param_dist = {
        'n_estimators': [80, 120, 160],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.03, 0.08, 0.15],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_base = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_search = RandomizedSearchCV(xgb_base, xgb_param_dist, n_iter=6, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)

    models_to_train = {
        "Logistic Regression": lr_search,
        "Random Forest": rf_search,
        "XGBoost": xgb_search
    }

    comparison_results = []
    trained_models = {}

    for name, search in models_to_train.items():
        print(f"\nTraining and tuning {name}...")
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        trained_models[name] = best_est

        # Predictions on held-out test set
        test_preds = best_est.predict(X_test)
        test_probs = best_est.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, test_probs)
        pr_auc = average_precision_score(y_test, test_probs)

        print(f"[{name}] Best CV ROC-AUC: {search.best_score_:.4f} | Test ROC-AUC: {roc_auc:.4f} | Test F1: {f1:.4f} | Test Recall: {rec:.4f}")

        comparison_results.append({
            "Model": name,
            "Best_CV_ROC_AUC": round(float(search.best_score_), 4),
            "Test_Accuracy": round(float(acc), 4),
            "Test_Precision": round(float(prec), 4),
            "Test_Recall": round(float(rec), 4),
            "Test_F1": round(float(f1), 4),
            "Test_ROC_AUC": round(float(roc_auc), 4),
            "Test_PR_AUC": round(float(pr_auc), 4),
            "Best_Parameters": str(search.best_params_)
        })

    df_comp = pd.DataFrame(comparison_results)
    return trained_models, df_comp

def run_training_pipeline(data_csv: str = None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_csv is None:
        data_csv = os.path.join(base_dir, "data", "processed", "cleaned_resumes.csv")
    if not os.path.exists(data_csv):
        raw_csv = os.path.join(base_dir, "data", "raw", "resumes_15000.csv")
        if not os.path.exists(raw_csv):
            from src.generate_dataset import main as gen_main
            gen_main()
        from src.cleaning import run_cleaning_pipeline
        run_cleaning_pipeline()

    print("=" * 65)
    print("AI Candidate Scoring: Training & Evaluation Pipeline")
    print("=" * 65)

    df = pd.read_csv(data_csv)
    y = df['shortlisted'].values

    # Step 5 & 8: Isolate Sensitive Attributes and Split
    df_features, df_sensitive = isolate_sensitive_attributes(df)

    print(f"Total dataset: {df.shape[0]} rows.")
    print(f"Positive class (Shortlisted): {np.sum(y == 1)} ({np.mean(y == 1)*100:.2f}%)")

    # Stratified 80/20 train/test split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, stratify=y, random_state=42)

    X_train_df = df_features.iloc[train_idx].reset_index(drop=True)
    X_test_df = df_features.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    sensitive_train = df_sensitive.iloc[train_idx].reset_index(drop=True)
    sensitive_test = df_sensitive.iloc[test_idx].reset_index(drop=True)

    # Save sensitive sets aligned with train/test indices
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    sensitive_test.to_csv(os.path.join(proc_dir, "sensitive_test.csv"), index=False)
    sensitive_train.to_csv(os.path.join(proc_dir, "sensitive_train.csv"), index=False)

    # Fit Feature Pipeline ONLY on training features
    print("\nFitting feature pipeline on training set...")
    pipeline = CandidateFeaturePipeline(top_k_skills=40, tfidf_max_features=60)
    pipeline.fit(X_train_df)

    X_train = pipeline.transform(X_train_df)
    X_test = pipeline.transform(X_test_df)
    feature_names = pipeline.get_feature_names()

    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    print(f"Features encoded: {len(feature_names)}")

    # Step 7: Class Imbalance comparison
    df_imbalance = compare_imbalance_strategies(X_train, y_train)
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    df_imbalance.to_csv(os.path.join(reports_dir, "class_imbalance_comparison.csv"), index=False)

    # Step 9: Train and Compare Models
    trained_models, df_comp = tune_and_compare_models(X_train, y_train, X_test, y_test)
    df_comp.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)

    # Select Best Model based on composite of ROC-AUC and F1
    df_comp['Selection_Score'] = df_comp['Test_ROC_AUC'] * 0.6 + df_comp['Test_F1'] * 0.4
    best_row = df_comp.sort_values(by='Selection_Score', ascending=False).iloc[0]
    best_model_name = best_row['Model']
    best_model = trained_models[best_model_name]

    print(f"\n=======================================================")
    print(f"[BEST MODEL SELECTED]: {best_model_name}")
    print(f"Test ROC-AUC: {best_row['Test_ROC_AUC']} | Test F1: {best_row['Test_F1']} | Test PR-AUC: {best_row['Test_PR_AUC']}")
    print(f"=======================================================")

    # Export artifacts to models/
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(pipeline, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.joblib"))
    joblib.dump(pipeline.skill_encoder.top_skills_, os.path.join(models_dir, "top_skills.joblib"))

    # Save X_test, y_test for downstream evaluations
    joblib.dump(X_test, os.path.join(proc_dir, "X_test.joblib"))
    joblib.dump(y_test, os.path.join(proc_dir, "y_test.joblib"))
    joblib.dump(X_train, os.path.join(proc_dir, "X_train.joblib"))
    joblib.dump(y_train, os.path.join(proc_dir, "y_train.joblib"))

    # Save metadata.json
    metadata = {
        "training_date": datetime.now().isoformat(),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features_count": int(len(feature_names)),
        "best_model": best_model_name,
        "metrics": {
            "accuracy": float(best_row['Test_Accuracy']),
            "precision": float(best_row['Test_Precision']),
            "recall": float(best_row['Test_Recall']),
            "f1_score": float(best_row['Test_F1']),
            "roc_auc": float(best_row['Test_ROC_AUC']),
            "pr_auc": float(best_row['Test_PR_AUC'])
        },
        "feature_names": feature_names,
        "top_skills": pipeline.skill_encoder.top_skills_
    }
    with open(os.path.join(models_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Best model, preprocessor, and metadata saved to: {models_dir}")
    return best_model, pipeline, df_comp

if __name__ == "__main__":
    run_training_pipeline()
