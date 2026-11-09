# ⚖️ AI Candidate Scoring & Shortlisting System with Fairness Analysis and Explainable AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning talent intelligence and shortlisting framework designed to evaluate candidate resumes, calculate a standardized suitability score (0–100), rank applicants per job role, detect and mitigate demographic biases, and generate plain-English explainable AI (XAI) factors for recruiters.

---

> ### ⚠️ Mandatory Ethical & Operational Notice
> **DECISION-SUPPORT ONLY**: The goal of this system is to predict whether a candidate should be shortlisted (`1`) or not (`0`) for a role, output a continuous suitability score from `0` to `100`, and rank candidates for talent acquisition teams. **The system is strictly designed for DECISION SUPPORT and must never be used as an automated final decision or sole basis for hiring.** All outputs require qualified human recruiter review.

---

## 📋 System Architecture

```text
candidate-scoring/
├── data/
│   ├── raw/resumes_15000.csv              # 15,000 synthetic candidate resumes with historical bias
│   ├── processed/cleaned_resumes.csv      # Standardized, cleaned, deduplicated dataset
│   ├── processed/sensitive_test.csv       # Isolated protected attributes for fairness audits
├── src/
│   ├── generate_dataset.py                # Synthetic dataset generation (seed=42)
│   ├── resume_parser.py                   # Structured text/PDF/DOCX extraction with skill synonyms
│   ├── cleaning.py                        # Casing normalization, experience parsing, IQR capping
│   ├── features.py                        # Pipeline encoding: ordinal, one-hot, multi-hot skills, TF-IDF
│   ├── eda.py                             # Exploratory data analysis & historical disparity charts
│   ├── train.py                           # 5-fold CV, imbalance comparison, RandomizedSearchCV tuning
│   ├── evaluate.py                        # Confusion matrix, ROC, PR, calibration curves, metrics.json
│   ├── fairness.py                        # Fairlearn audit (EEOC 80% rule, Equalized Odds mitigation)
│   ├── explain.py                         # SHAP beeswarm/dependence, LIME concordance, recruiter text
│   ├── score.py                           # 0-100 score conversion, match tiers, per-role ranking
├── models/                                # Trained XGBoost, preprocessor, feature names, metadata.json
├── reports/                               # Visualizations, metrics.json, fairness_report.json
│   ├── eda/                               # EDA charts and eda_summary.md
│   ├── fairness/                          # Selection rate and disparate impact plots
│   ├── explainability/                    # SHAP summary, bar plots, dependence plots
├── app/
│   ├── streamlit_app.py                   # Full interactive 6-tab Streamlit dashboard
├── tests/
│   ├── test_pipeline.py                   # Comprehensive unit & integration pytest suite
├── sample_resumes/                        # 10 diverse test resumes (.txt/.pdf)
├── docs/
│   ├── ETHICS_AND_FAIRNESS.md             # In-depth algorithmic fairness, legal & governance charter
│   ├── model_card.md                      # Detailed system model card & performance benchmarks
├── run_pipeline.py                        # Single-command CLI runner for full end-to-end pipeline
├── requirements.txt                       # Pinned cloud-ready dependencies
├── README.md                              # Setup, execution, and deployment guide
└── .streamlit/config.toml                 # Modern UI configuration
```

---

## 🚀 Quickstart & Local Installation

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/candidate-scoring.git
cd candidate-scoring

# Create virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline via Single CLI Command
To generate the dataset, clean, engineer features, train, evaluate, audit fairness, and run tests sequentially:
```bash
python run_pipeline.py
```

### 4. Or Run Individual Stages Step-by-Step
```bash
# Step 1: Generate 15,000 synthetic candidate resumes
python src/generate_dataset.py

# Step 2: Clean data and parse strings
python src/cleaning.py

# Step 3: Run feature engineering and preprocessor fit
python src/features.py

# Step 4: Run exploratory data analysis
python src/eda.py

# Step 5: Train and tune models (Logistic Reg vs RF vs XGBoost)
python src/train.py

# Step 6: Evaluate test accuracy, ROC, PR, and calibration
python src/evaluate.py

# Step 7: Audit algorithmic fairness and run post-processing mitigation
python src/fairness.py

# Step 8: Generate global SHAP and LIME explainability plots
python src/explain.py

# Step 9: Launch the interactive Streamlit application
streamlit run app/streamlit_app.py
```

---

## 📊 Key Results Summary

### 1. Model Benchmark Comparison
*Evaluated on 3,000 unseen holdout candidate records (20% stratified test split):*

| Model Architecture | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Accuracy | Test Recall | Test Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 0.835 | 0.842 | 0.715 | 0.618 | 78.4% | 76.2% | 52.1% |
| **Random Forest (Balanced)** | 0.865 | 0.871 | 0.762 | 0.655 | 81.2% | 73.8% | 58.9% |
| **XGBoost (Selected Best)** | **0.878** | **0.885** | **0.784** | **0.678** | **82.5%** | **74.5%** | **62.3%** |

### 2. Class Imbalance Strategy Comparison
- **None (Unweighted)**: High accuracy but low recall on positive class (~54%).
- **SMOTE (Training Fold Only)**: Improved recall (~75%) but increased false positives.
- **`class_weight='balanced'` / `scale_pos_weight` (Winner)**: Optimal trade-off with peak F1-score (~0.68) and ROC-AUC without synthetic artifact distortion.

### 3. Algorithmic Fairness & Mitigation (Four-Fifths Rule)
- **Raw Historical Gender Disparate Impact**: $0.72$ (Adverse impact flagged: Female selection rate $20.2\%$ vs Male $28.1\%$).
- **After Equalized Odds Mitigation (`ThresholdOptimizer`)**: Disparate Impact Ratio elevated to **$0.86$** (Passes EEOC $80\%$ rule), closing the Equal Opportunity TPR gap from $0.12$ to $0.03$ with only a minor $1.6\%$ trade-off in overall accuracy.

---

## 🖥️ Streamlit Application Walkthrough

The web interface is organized into 6 interactive modules:
1. **Single Resume Scoring**: Upload a `.pdf`, `.docx`, or `.txt` resume or paste raw text. The parser extracts fields into an editable form, computes the 0–100 suitability score on a Plotly gauge, assigns a match tier, and provides a SHAP waterfall plot and plain-English explanation.
2. **Batch Candidate Ranking**: Upload a multi-candidate CSV or evaluate pre-loaded sample applicants. Filter by job role and minimum score, inspect rankings, and download the curated shortlist as CSV.
3. **Model Performance**: Interactive performance cards, confusion matrix, ROC curve, precision-recall curve, and calibration reliability diagrams.
4. **Fairness Dashboard**: Disparate impact gauges across Gender, Age Groups, and College Tiers with automated red/green compliance indicators and before/after mitigation tables.
5. **Explainability**: Global SHAP beeswarm summary plots, mean absolute feature importances, and dependence plots for top decision drivers.
6. **About & Ethics**: Operational guidelines, GDPR / India DPDP regulatory alignments, and human-in-the-loop protocols.

---

## 🧪 Automated Testing
Run the complete test suite with `pytest`:
```bash
pytest tests/ -v
```
Verifies:
- Text, PDF, and DOCX parser accuracy and skill synonym resolution.
- Data cleaning deduplication, degree/city standardization, and IQR outlier capping.
- Feature pipeline input-output dimensions and absence of sensitive attributes.
- Scoring calibration, match tier assignment, and role ranking.
- Fairlearn disparate impact and demographic parity calculation functions.

---

## ☁️ Step-by-Step Deployment Instructions (Streamlit Community Cloud)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI Candidate Scoring System with Fairness & XAI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
   *(Note: Model files in `models/` total less than 3 MB, comfortably below GitHub's 100 MB file limit).*

2. **Deploy on Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"New App"**.
   - Select your repository, branch (`main`), and set Main file path to:
     ```text
     app/streamlit_app.py
     ```
   - Click **"Deploy!"**.
   - *Automatic Fallback*: If model files are not present in a new container, the application automatically detects this and triggers the pipeline training on first run.

---

## ⚖️ Ethical & Legal Compliance Charter
For a comprehensive analysis of proxy variable risks, transparency measures, candidate right to explanation, and GDPR/DPDP alignment, read:
👉 **[docs/ETHICS_AND_FAIRNESS.md](docs/ETHICS_AND_FAIRNESS.md)**

