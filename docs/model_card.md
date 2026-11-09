# Model Card: AI Candidate Scoring & Shortlisting System

## 1. Model Details
- **Model Name:** Candidate Suitability & Shortlisting Classifier
- **Model Versions Evaluated:** Logistic Regression (L2-regularized), Random Forest Classifier, XGBoost (Gradient Boosted Trees).
- **Primary Frameworks:** Scikit-Learn (1.4+), XGBoost (2.0+), Imbalanced-Learn, Fairlearn, SHAP, LIME.
- **Model Type:** Binary Supervised Classification with continuous probability output mapped to a 0–100 suitability score.
- **License:** MIT Open Source
- **Contact:** Talent AI Engineering & Governance Team

---

## 2. Intended Use
- **Primary Intended Use:** Algorithmic decision-support tool designed for recruiters and talent acquisition specialists to triage large applicant pools, rank candidate resumes against role requirements, and highlight key qualification factors.
- **Intended Users:** Corporate recruiters, hiring managers, talent acquisition coordinators.
- **Decision Boundary:** Decision **SUPPORT** only. All candidates flagged as "Possible Match" or "Strong Match" must undergo human interview and evaluation. Autonomous candidate rejection without human oversight is strictly prohibited.

---

## 3. Out-of-Scope & Prohibited Uses
- **Prohibited:** Autonomous, unmonitored candidate rejection or final hiring termination decisions.
- **Prohibited:** Cross-domain transfer (e.g., using this model for credit scoring, insurance underwriting, or academic admissions).
- **Prohibited:** Evaluation of resumes without prior disclosure to applicants regarding the use of AI evaluation tools.

---

## 4. Factors & Features
The model consumes a multidimensional feature vector derived from candidate resumes:
- **Merit Features (Continuous & Ordinal):**
  - `role_skill_match_score`: Synthesized percentage of role-specific requirements satisfied.
  - `years_experience`: Normalized professional industry experience.
  - `education_level_ordinal`: Ordinal hierarchy (High School < Diploma < Bachelors < Masters < PhD).
  - `gpa_or_percentage`: Standardized academic performance (10-point scale).
  - `projects_count`: Number of verified technical / production projects.
  - `certifications_count`: Count of recognized industry credentials.
  - `publications_count`: Research publications in recognized conferences/journals.
  - `employment_gap_months`: Total duration of career breaks.
- **Categorical Features (One-Hot Encoded):**
  - `degree_field`: Discipline of study (Computer Science, IT, Data Science, Electronics, Mechanical, Commerce, Arts).
  - `job_role_applied`: Target role taxonomy.
  - `current_company_type`: Prior employer category (Product, MNC, Service, Startup, None).
- **Multi-Hot Skills & NLP:**
  - Binary indicators across the top 40 industry technical skills.
  - TF-IDF unigram and bigram features extracted from candidate summary paragraphs.

---

## 5. Protected Attributes & Fairness Safeguards
- **Excluded Sensitive Attributes:** `gender`, `age`, `name`, and `candidate_id` are permanently excluded from the feature space.
- **Proxy Mitigation:** Experience is bounded with non-linear diminishing returns to prevent age proxying.
- **Fairness Benchmarking:** All models are subjected to algorithmic bias audits using Fairlearn across Gender, Age Groups, and College Tiers.
- **Mitigation Architecture:** Fairlearn's `ThresholdOptimizer` applies equalized odds post-processing to enforce compliance with the EEOC Four-Fifths (80%) Rule ($DI \ge 0.80$).

---

## 6. Performance & Quantitative Results
*Trained on 12,000 stratified candidate records; evaluated on 3,000 held-out test candidates:*

| Evaluation Metric | Baseline Logistic Reg | Tuned Random Forest | Tuned XGBoost (Selected) |
| :--- | :--- | :--- | :--- |
| **Test ROC-AUC** | ~0.842 | ~0.871 | **~0.885** |
| **Test PR-AUC** | ~0.715 | ~0.762 | **~0.784** |
| **Test F1-Score** | ~0.618 | ~0.655 | **~0.678** |
| **Test Accuracy** | ~78.4% | ~81.2% | **~82.5%** |
| **Test Recall** | ~76.2% | ~73.8% | **~74.5%** |
| **Test Precision** | ~52.1% | ~58.9% | **~62.3%** |

---

## 7. Explainability & Interpretability
Every score generated includes:
1. **Global Feature Impact:** SHAP summary beeswarm and feature importance rankings indicating macro drivers (Role Skill Match, Experience, Education, Projects).
2. **Local Candidate Explanations:** SHAP waterfall decompositions showing positive and negative points for each applicant.
3. **Dual Explanation Concordance:** Evaluated against local LIME tabular models to confirm stability.
4. **Natural Language Generation:** Automated recruiter narratives articulating the reasoning behind each match tier assignment.

---

## 8. Limitations & Environmental Constraints
- **Synthetic Training Origin:** Trained on a controlled synthetic distribution (15,000 candidates). While mathematically realistic, real-world resume formats, non-traditional career paths, and emerging technologies may require continuous re-training.
- **Skill Keyword Reliance:** Although fuzzy synonyms are supported, atypical formatting or idiosyncratic project descriptions may underestimate candidate competency. Recruiter review is essential.

