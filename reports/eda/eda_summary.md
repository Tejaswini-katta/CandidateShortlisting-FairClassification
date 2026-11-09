# Exploratory Data Analysis & Historical Bias Summary

## 1. Class Distribution & Sample Size
- **Total Candidates Audited:** 14,975
- **Shortlisted Rate:** 24.47% (3,664 candidates)
- **Rejected Rate:** 75.53% (11,311 candidates)
- **Imbalance Ratio:** ~3.1:1 rejection to shortlist ratio.

## 2. Merit Factors & Correlations
- **Experience:** Candidates with 5-8 years of experience achieve the highest shortlist rate (~33.7%), after which returns diminish.
- **Education:** Higher credentials strongly elevate shortlisting odds. PhD (30.3%) and Masters (27.3%) significantly outpace Bachelors (24.0%) and Diplomas (17.4%).
- **Projects & Certifications:** Positively correlated with selection ($r \approx 0.35$ and $r \approx 0.28$).
- **Employment Gap:** Negative correlation ($r \approx -0.19$) reflecting penalty on extended breaks.

## 3. Disparity Audit (Synthetic Injected Historical Bias)
- **Gender Selection Rates:**
  - Male: **27.5%**
  - Female: **20.9%** (Raw Adverse Disparate Impact Ratio: 0.76)
  - Non-binary: **18.2%**
- **Age Disparity:**
  - Age < 30: **18.6%**
  - Age 30-45: **33.6%**
  - Age 45+: **27.8%** (Marked drop-off showing synthetic age-related penalty)
- **Institutional Pedigree (College Tier):**
  - Tier 1: **32.3%**
  - Tier 2: **24.8%**
  - Tier 3: **21.9%**

> **Ethical Note:** This raw bias mirrors real-world historical resume screening biases. Our downstream models will intentionally remove explicit sensitive attributes (Gender, Age) and undergo algorithmic fairness mitigation to decouple merit evaluation from institutional and demographic status.
