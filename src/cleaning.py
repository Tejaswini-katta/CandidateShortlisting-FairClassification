"""
Data Cleaning and Preprocessing Module
======================================
Standardizes degree names, city names, casing, parses string experience numbers,
removes duplicate records, handles missing values (with missingness indicators),
caps statistical outliers using IQR, and logs transformations to reports/cleaning_log.txt.
"""

import os
import re
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

DEGREE_MAPPING = {
    r'.*(cs|computer\s*science|cse|software).*': 'Computer Science',
    r'.*(information\s*technology|it|infotech).*': 'Information Technology',
    r'.*(data\s*science|artificial\s*intelligence|ai|ds).*': 'Data Science',
    r'.*(electronic|ece|electrical|eee|telecom).*': 'Electronics',
    r'.*(mechanical|mech|civil|automobile).*': 'Mechanical',
    r'.*(commerce|finance|accounting|economics|bcom|b\.com).*': 'Commerce',
    r'.*(art|humanities|business|bba|management|mba).*': 'Arts'
}

CITY_MAPPING = {
    r'.*(bengaluru|bangalore).*': 'Bengaluru',
    r'.*(mumbai|bombay).*': 'Mumbai',
    r'.*(delhi|gurgaon|gurugram|noida|ncr).*': 'Delhi NCR',
    r'.*(pune).*': 'Pune',
    r'.*(hyderabad).*': 'Hyderabad',
    r'.*(chennai|madras).*': 'Chennai',
    r'.*(kolkata|calcutta).*': 'Kolkata',
    r'.*(ahmedabad).*': 'Ahmedabad',
    r'.*(jaipur).*': 'Jaipur',
    r'.*(kochi|cochin).*': 'Kochi'
}

def clean_degree(val: str) -> str:
    """Standardize degree field text into canonical categories."""
    if pd.isna(val) or not str(val).strip():
        return "Computer Science"
    v_str = str(val).lower().strip()
    for pattern, canonical in DEGREE_MAPPING.items():
        if re.match(pattern, v_str):
            return canonical
    return "Computer Science"

def clean_city(val: str) -> str:
    """Standardize city names into canonical Indian metropolitan names."""
    if pd.isna(val) or not str(val).strip():
        return "Bengaluru"
    v_str = str(val).lower().strip()
    for pattern, canonical in CITY_MAPPING.items():
        if re.match(pattern, v_str):
            return canonical
    return str(val).strip().title()

def parse_experience(val: any) -> float:
    """Parse experience values which may be numeric or formatted strings like '5 yrs'."""
    if pd.isna(val):
        return 2.0
    if isinstance(val, (int, float)):
        return float(val)
    v_str = str(val).lower().strip()
    match = re.search(r'([0-9]+(?:\.[0-9]+)?)', v_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 2.0
    return 2.0

def clean_skills_string(skills_str: any) -> str:
    """Normalize comma-separated skill string: casing, whitespace, and duplicates."""
    if pd.isna(skills_str) or not str(skills_str).strip():
        return "Communication, Problem Solving"
    tokens = [s.strip().title() for s in str(skills_str).split(",") if s.strip()]
    # Deduplicate while preserving order
    seen = set()
    cleaned = []
    for t in tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            cleaned.append(t)
    return ", ".join(cleaned)

def clean_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cleans candidate dataframe:
    - Removes duplicate rows
    - Parses and coerces numeric types
    - Standardizes categorical values
    - Imputes missing values and creates missing indicators
    - Caps outliers via IQR
    - Returns cleaned DataFrame and detailed change log entries
    """
    df = df_raw.copy()
    log = []
    log.append(f"Initial raw record count: {len(df)}")

    # 1. Deduplication
    dup_mask = df.duplicated(subset=['name', 'age', 'job_role_applied', 'resume_text'], keep='first')
    dup_count = dup_mask.sum()
    if dup_count > 0:
        df = df[~dup_mask].copy()
        log.append(f"Removed {dup_count} duplicate records. New record count: {len(df)}")
    else:
        log.append("No duplicate records found.")

    # 2. Parse and coerce 'years_experience'
    log.append("Standardizing and parsing 'years_experience' from string/numeric format.")
    df['years_experience'] = df['years_experience'].apply(parse_experience)

    # 3. Standardize Degree Fields and Cities
    log.append("Standardizing degree fields to canonical categories.")
    df['degree_field'] = df['degree_field'].apply(clean_degree)

    log.append("Standardizing city and location names.")
    df['location'] = df['location'].apply(clean_city)

    # 4. Standardize Skills strings
    log.append("Normalizing casing and spacing for skills strings.")
    df['skills'] = df['skills'].apply(clean_skills_string)

    # 5. Handle Missing Values
    # Missing GPA
    gpa_missing_count = df['gpa_or_percentage'].isnull().sum()
    if gpa_missing_count > 0:
        median_gpa = df['gpa_or_percentage'].median()
        df['gpa_is_missing'] = df['gpa_or_percentage'].isnull().astype(int)
        df['gpa_or_percentage'] = df['gpa_or_percentage'].fillna(median_gpa)
        log.append(f"Imputed {gpa_missing_count} missing 'gpa_or_percentage' with median ({median_gpa:.2f}). Added 'gpa_is_missing' indicator.")
    else:
        df['gpa_is_missing'] = 0

    # Missing College Tier
    tier_missing_count = df['college_tier'].isnull().sum()
    if tier_missing_count > 0:
        mode_tier = df['college_tier'].mode()[0] if not df['college_tier'].mode().empty else "Tier 2"
        df['college_tier'] = df['college_tier'].fillna(mode_tier)
        log.append(f"Imputed {tier_missing_count} missing 'college_tier' with mode ('{mode_tier}').")

    # Missing Current Company Type
    comp_missing_count = df['current_company_type'].isnull().sum()
    if comp_missing_count > 0:
        mode_comp = df['current_company_type'].mode()[0] if not df['current_company_type'].mode().empty else "MNC"
        df['current_company_type'] = df['current_company_type'].fillna(mode_comp)
        log.append(f"Imputed {comp_missing_count} missing 'current_company_type' with mode ('{mode_comp}').")

    # Missing Expected Salary
    salary_missing_count = df['expected_salary_lpa'].isnull().sum()
    if salary_missing_count > 0:
        med_salary = df.groupby('job_role_applied')['expected_salary_lpa'].transform('median')
        df['expected_salary_lpa'] = df['expected_salary_lpa'].fillna(med_salary).fillna(df['expected_salary_lpa'].median())
        log.append(f"Imputed {salary_missing_count} missing 'expected_salary_lpa' with role-specific median.")

    # 6. Outlier Capping via IQR
    def cap_outliers_iqr(series: pd.Series, lower_bound_min: float = None, upper_bound_max: float = None) -> Tuple[pd.Series, int]:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        if lower_bound_min is not None:
            low = max(low, lower_bound_min)
        if upper_bound_max is not None:
            high = min(high, upper_bound_max)
        capped = series.clip(lower=low, upper=high)
        n_outliers = (series < low).sum() + (series > high).sum()
        return capped, n_outliers

    # Cap GPA
    # Scale correction if percentage entered as >10
    high_gpa_mask = df['gpa_or_percentage'] > 10.0
    if high_gpa_mask.sum() > 0:
        df.loc[high_gpa_mask, 'gpa_or_percentage'] = df.loc[high_gpa_mask, 'gpa_or_percentage'] / 10.0
        log.append(f"Rescaled {high_gpa_mask.sum()} entries with GPA > 10.0 to 10-point scale.")

    df['gpa_or_percentage'], gpa_outliers = cap_outliers_iqr(df['gpa_or_percentage'], lower_bound_min=5.0, upper_bound_max=10.0)
    log.append(f"Capped {gpa_outliers} GPA outliers within bounds [5.0, 10.0].")

    # Cap Experience (cannot exceed age - 18)
    age_cap_mask = df['years_experience'] > (df['age'] - 18)
    if age_cap_mask.sum() > 0:
        df.loc[age_cap_mask, 'years_experience'] = (df.loc[age_cap_mask, 'age'] - 21).clip(lower=0)
        log.append(f"Fixed {age_cap_mask.sum()} inconsistent experience vs age records.")

    df['years_experience'], exp_outliers = cap_outliers_iqr(df['years_experience'], lower_bound_min=0.0, upper_bound_max=35.0)
    log.append(f"Capped {exp_outliers} experience outliers within bounds [0.0, 35.0].")

    # Cap Expected Salary
    df['expected_salary_lpa'], sal_outliers = cap_outliers_iqr(df['expected_salary_lpa'], lower_bound_min=2.5, upper_bound_max=75.0)
    log.append(f"Capped {sal_outliers} expected salary outliers within bounds [2.5, 75.0] LPA.")

    # 7. Type Validation
    df['certifications_count'] = df['certifications_count'].fillna(0).astype(int)
    df['projects_count'] = df['projects_count'].fillna(0).astype(int)
    df['publications_count'] = df['publications_count'].fillna(0).astype(int)
    df['employment_gap_months'] = df['employment_gap_months'].fillna(0).astype(int)
    df['notice_period_days'] = df['notice_period_days'].fillna(30).astype(int)
    df['shortlisted'] = df['shortlisted'].astype(int)

    log.append(f"Final cleaned dataset shape: {df.shape}")
    log.append("Data cleaning completed successfully.")

    return df, log

def run_cleaning_pipeline(input_csv: str = None, output_csv: str = None, log_file: str = None) -> pd.DataFrame:
    """Run end-to-end cleaning and write output and logs."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if input_csv is None:
        input_csv = os.path.join(base_dir, "data", "raw", "resumes_15000.csv")
    if output_csv is None:
        output_csv = os.path.join(base_dir, "data", "processed", "cleaned_resumes.csv")
    if log_file is None:
        log_file = os.path.join(base_dir, "reports", "cleaning_log.txt")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    print(f"Reading raw data from: {input_csv}")
    df_raw = pd.read_csv(input_csv)

    df_cleaned, log_lines = clean_dataset(df_raw)

    df_cleaned.to_csv(output_csv, index=False)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"[SUCCESS] Cleaned data saved to: {output_csv}")
    print(f"Cleaning log written to: {log_file}")
    for line in log_lines:
        print(f"  * {line}")

    return df_cleaned

if __name__ == "__main__":
    run_cleaning_pipeline()

