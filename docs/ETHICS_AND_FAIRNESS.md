# Ethics, Algorithmic Fairness & Governance Framework

> **CRITICAL NOTICE: DECISION SUPPORT SYSTEM ONLY**  
> This artificial intelligence system is strictly engineered as an assistive decision-support tool for human talent acquisition professionals. It **must never** be deployed as an autonomous or final decision-maker in employment candidate shortlisting or hiring workflows. All shortlisting outcomes require qualified human review, contextual validation, and final accountability.

---

## 1. Synthetic Data Disclaimer & Injected Historical Bias
This repository uses a synthetic dataset of 15,000 candidate profiles generated algorithmically with a fixed random seed (`42`). To reflect real-world algorithmic hiring hazards and demonstrate the detection capabilities of algorithmic fairness frameworks:
1. **Merit-First Latent Probability**: Selection probability is primarily driven by objective qualifications: skill alignment with job role taxonomy, relevant work experience, highest degree earned, academic GPA, verified technical projects, and industry certifications.
2. **Deliberate Synthetic Bias**: Mild historical hiring skews were intentionally injected:
   - A mathematical penalty on career breaks / employment gaps $>6$ months ($\Delta \approx -0.35$ to $-0.65$ log-odds).
   - An institutional prestige premium for Tier 1 universities ($\Delta \approx +0.35$ log-odds).
   - A slight demographic preference favoring male applicants over female and non-binary applicants ($\Delta \approx +0.22$ vs $-0.15$ log-odds).
   - An age penalty on applicants over 45 years of age ($\Delta \approx -0.40$ log-odds).
3. **Purpose**: These synthetic distortions are strictly illustrative. They allow recruiters, data scientists, and compliance auditors to observe how unconstrained machine learning models amplify historical human bias and verify how post-processing mitigation techniques restore equity.

---

## 2. Sensitive Attributes & Feature Isolation
To uphold the principle of **Fairness Through Unawareness** as a first line of defense, all direct protected demographic identifiers are strictly quarantined:
- **Excluded Features**: `gender`, `age`, `name`, and `candidate_id` are completely stripped from model inputs and preprocessing pipelines.
- **Auditing Vault**: Sensitive attributes are isolated into a separate validation partition (`sensitive_test.csv`) strictly used by compliance scripts (`src/fairness.py`) and never exposed during model training or inference scoring.

---

## 3. Proxy Variable Risks & Auditing
Excluding explicit demographic columns does not guarantee fairness if other input variables encode demographic signals. We routinely audit potential proxy variables:
- **Experience as an Age Proxy**: Years of professional experience naturally correlates with chronological age ($r \approx 0.82$). To prevent experience from functioning as an illicit age screen, non-linear capping and diminishing return bounds are applied ($>6$ years), ensuring senior candidates are evaluated on skill depth rather than penalizing elder applicants.
- **Institutional Pedigree (College Tier)**: University tiers frequently correlate with socioeconomic status and geographic mobility. College tier is audited for disparate impact across regions.
- **Degree Fields**: Cross-tabulation audits monitor whether STEM disciplines exhibit gender imbalances that could inadvertently leak into model predictions.

---

## 4. Algorithmic Fairness Metrics & Benchmarks
Our fairness auditing pipeline leverages Fairlearn's `MetricFrame` across three primary sensitive dimensions: **Gender**, **Age Group** ($<30$, $30-45$, $45+$), and **College Tier** (Tier 1, Tier 2, Tier 3).

| Metric | Mathematical Definition | Compliance Standard | Target Interpretation |
| :--- | :--- | :--- | :--- |
| **Selection Rate ($SR$)** | $P(\hat{Y}=1 \mid A=a)$ | N/A | Proportion of group candidates shortlisted |
| **Demographic Parity Diff** | $\max_a SR_a - \min_a SR_a$ | $\le 0.10$ | Difference in positive selection rates between groups |
| **Disparate Impact Ratio** | $\frac{\min_a SR_a}{\max_a SR_a}$ | $\ge 0.80$ | **EEOC Four-Fifths (80%) Rule**: Ratios below 0.80 flag adverse impact |
| **Equal Opportunity Diff** | $\max_a TPR_a - \min_a TPR_a$ | $\le 0.10$ | Gap in True Positive Rate (qualified candidates selected equally) |
| **Equalized Odds Diff** | $\max(\Delta TPR, \Delta FPR)$ | $\le 0.10$ | Parity in both true and false positive rates across demographics |

---

## 5. Algorithmic Mitigation & The Utility-Fairness Trade-Off
When baseline models inherit historical skews, post-processing mitigation is applied using Fairlearn's `ThresholdOptimizer` under **Equalized Odds** constraints:
- **Mechanism**: The optimizer calculates demographic-group-specific threshold boundaries that align false positive and true positive rates without modifying the underlying feature representations.
- **The Trade-Off**:
  - Baseline XGBoost achieves higher nominal Test Accuracy (~82%) and F1 (~0.68), but exhibits an adverse Disparate Impact Ratio on protected demographics ($\approx 0.72$).
  - Post-mitigation thresholding successfully elevates the Disparate Impact Ratio above the legally recognized $0.80$ threshold and narrows the Equal Opportunity gap to $<0.05$.
  - This fairness enhancement incurs a modest, acceptable reduction in overall accuracy (~1.5% to 2.5%), demonstrating a transparent, principled trade-off between raw predictive accuracy and systemic equity.

---

## 6. Transparency, Interpretability & Explainable AI (XAI)
Automated scoring models must never function as black boxes. Every prediction generated by this system includes:
1. **Global SHAP Explanations**: Beeswarm summary plots and mean absolute SHAP bar charts documenting the macro-level drivers of the model.
2. **Local Interpretability**: Individual SHAP waterfall charts revealing the exact positive and negative pushes behind a candidate's score.
3. **LIME Dual-Verification**: Tabular LIME explanations are computed alongside SHAP to verify model fidelity.
4. **Recruiter Natural Language Summaries**: Automatically synthesizes plain-English explanations (e.g., *"Scored 84/100 (Strong Match) for Data Scientist due to 6 years experience, 90% skill match, and Masters degree; constrained by a 12-month employment gap"*).

---

## 7. Legal, Regulatory & Privacy Compliance
This architecture adheres to international data protection and AI governance frameworks:
- **India Digital Personal Data Protection (DPDP) Act 2023**: Implements strict data minimization, purpose limitation, and storage limitation. Candidate resumes and extracted tabular profiles are processed locally with zero third-party cloud data leakage.
- **EU General Data Protection Regulation (GDPR - Article 22)**: Prohibits solely automated individual decision-making with legal or similarly significant effects without explicit human intervention. This system complies by functioning exclusively as a recommendation mechanism.
- **New York City Local Law 144 (Automated Employment Decision Tools)**: Requires annual independent bias audits, publicly posted summary statistics of selection rates, and candidate notification regarding algorithmic tools.

---

## 8. Mitigation of Automation Bias & Human Oversight
- **Automation Bias**: Human recruiters tend to over-rely on algorithmic rankings and rubber-stamp machine outputs.
- **Safeguards Implemented**:
  1. Permanent visual banners in the Streamlit user interface reminding recruiters of human accountability.
  2. Granular, editable feature extraction fields allowing recruiters to correct parser errors before scoring.
  3. Presentation of scores as continuous suitability ranges and confidence match tiers rather than binary pass/fail verdicts.
  4. Mandatory manual review protocols for all candidates categorized as "Possible Match".

---

## 9. Candidate Rights: Explanation & Appeal
Every candidate evaluated through this system holds the fundamental right to:
1. **Transparent Notification**: Prior notice that algorithmic screening assistance is utilized during the initial application triage.
2. **Meaningful Explanation**: Access to the top positive and negative factors influencing their score breakdown.
3. **Formal Human Appeal**: The right to request an immediate, manual re-evaluation by a human talent acquisition specialist if they suspect parsing inaccuracies or unfair evaluation.

---

## 10. Operational Recommendations for Real-World Deployment
Before any organization attempts to adapt this system for real-world hiring:
1. **Continuous Bias Monitoring**: Implement drift and fairness monitoring jobs every quarter to detect shifts in applicant demographics or scoring distributions.
2. **Disparate Impact Threshold Alerts**: Automatically halt algorithmic filtering if the 4/5ths Disparate Impact ratio dips below $0.80$.
3. **Legal & Compliance Review**: Subject job role skill profiles and weighting logic to labor counsel to verify that all evaluation criteria satisfy job-related business necessity standards.

