"""
Synthetic Resume Dataset Generator (15,000 Candidates)
======================================================
Generates a realistic candidate dataset with ground-truth shortlisting logic
based on merit (skills, experience, education, GPA, projects, certifications)
plus injected mild historical bias (employment gaps, college tier, gender, age)
for fairness auditing and explainable AI demonstrations.
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker

def generate_candidates(n_samples: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Generate 15,000 synthetic candidate resumes with realistic fields,
    mild synthetic bias, and realistic data cleanliness issues."""
    np.random.seed(seed)
    random.seed(seed)
    fake = Faker('en_IN')
    Faker.seed(seed)

    # Roles and their target core skill profiles
    role_skill_map = {
        "Data Scientist": ["Python", "Machine Learning", "SQL", "Statistics", "Deep Learning", "Pandas", "Scikit-Learn", "NLP", "Data Visualization", "AWS"],
        "ML Engineer": ["Python", "Machine Learning", "Deep Learning", "Docker", "AWS", "MLOps", "PyTorch", "Kubernetes", "Git", "SQL"],
        "Data Analyst": ["SQL", "Excel", "Power BI", "Python", "Tableau", "Data Visualization", "Statistics", "Communication", "Reporting", "Pandas"],
        "Software Engineer": ["Java", "Python", "Docker", "AWS", "Git", "REST APIs", "SQL", "Data Structures", "System Design", "CI/CD"],
        "Business Analyst": ["Excel", "SQL", "Power BI", "Agile", "Communication", "Requirements Gathering", "Tableau", "Problem Solving", "Jira", "Process Mapping"]
    }

    all_skill_pool = list(set(skill for skills in role_skill_map.values() for skill in skills) | {
        "React", "Node.js", "C++", "Linux", "MongoDB", "Spark", "TensorFlow", "Leadership", "Flask", "Django", "FastAPI"
    })

    roles = list(role_skill_map.keys())
    role_weights = [0.24, 0.18, 0.22, 0.24, 0.12]

    indian_cities = [
        "Bengaluru", "Hyderabad", "Pune", "Mumbai", "Delhi NCR",
        "Chennai", "Kolkata", "Ahmedabad", "Gurugram", "Noida", "Jaipur", "Kochi"
    ]
    city_weights = [0.25, 0.18, 0.14, 0.12, 0.10, 0.08, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01]

    education_levels = ["High School", "Diploma", "Bachelors", "Masters", "PhD"]
    edu_weights = [0.03, 0.05, 0.62, 0.26, 0.04]

    degree_fields_clean = ["Computer Science", "Information Technology", "Electronics", "Data Science", "Mechanical", "Commerce", "Arts"]
    degree_weights = [0.40, 0.22, 0.14, 0.10, 0.06, 0.05, 0.03]

    college_tiers = ["Tier 1", "Tier 2", "Tier 3"]
    tier_weights = [0.15, 0.38, 0.47]

    company_types = ["Product", "MNC", "Service", "Startup", "None"]
    company_weights = [0.28, 0.32, 0.24, 0.12, 0.04]

    candidates = []

    # Resume text templates
    templates = [
        "Dynamic {role} professional with {exp:.1f} years of hands-on experience in building scalable solutions. Graduated with a {edu} in {degree} from a top-tier {tier} institution. Adept at {top_skills}. Proven track record of delivering {projects} impactful enterprise projects and holding {certs} recognized professional certifications. Committed to continuous learning, engineering excellence, and cross-functional team collaboration.",
        "Results-oriented candidate pursuing {role} opportunities with {exp:.1f} years working across {company_type} environments. Academic foundation: {edu} in {degree} (GPA {gpa:.2f}). Strong core competencies in {top_skills}. Authored {pubs} technical publications and spearheaded {projects} end-to-end industry implementations. Dedicated to applying state-of-the-art methodologies to solve high-impact organizational challenges.",
        "Passionate tech specialist with {exp:.1f} years of experience specializing as a {role}. Education: {edu} in {degree}. Proficient in {top_skills}. Known for rapid problem-solving, reliable execution, and delivering high quality code across {projects} production-level initiatives. Holds {certs} certifications and demonstrates strong proficiency in modern software and analytical stacks.",
        "Accomplished professional with {exp:.1f} years in technical roles, actively seeking a {role} position. Holding a {edu} in {degree} ({tier}), with a strong academic score of {gpa:.1f}. Core toolkit includes {top_skills}. Successfully completed {projects} projects. Demonstrated adaptability across fast-paced teams with {gap} months between previous assignments."
    ]

    for i in range(n_samples):
        cand_id = f"CAND-{i+1:05d}"
        
        # Gender distribution: ~56% Male, 41% Female, 3% Non-binary
        gender = np.random.choice(["Male", "Female", "Non-binary"], p=[0.56, 0.41, 0.03])
        if gender == "Male":
            name = fake.name_male()
        elif gender == "Female":
            name = fake.name_female()
        else:
            name = fake.name()

        # Age distribution: 21 to 58, median ~28
        age = int(np.clip(np.random.gamma(shape=9, scale=3.2), 21, 58))

        location = np.random.choice(indian_cities, p=city_weights)
        edu_level = np.random.choice(education_levels, p=edu_weights)
        deg_field = np.random.choice(degree_fields_clean, p=degree_weights)
        tier = np.random.choice(college_tiers, p=tier_weights)

        # Experience correlated with age and education
        min_age_for_exp = 21 if edu_level in ["High School", "Diploma", "Bachelors"] else (23 if edu_level == "Masters" else 26)
        max_possible_exp = max(0, age - min_age_for_exp)
        exp = round(float(np.random.uniform(0.0, max_possible_exp)), 1)
        if exp < 0.5 and max_possible_exp > 0:
            exp = 0.0

        # GPA/percentage: either scale of 10 (CGPA 6.0 to 9.9) or scale of 100 (60 to 98)
        # Standardized representation: CGPA out of 10 (e.g. 6.2 to 9.8)
        gpa = round(float(np.clip(np.random.normal(loc=7.8, scale=0.9), 5.5, 9.9)), 2)

        role = np.random.choice(roles, p=role_weights)
        comp_type = np.random.choice(company_types, p=company_weights)

        # Skills selection: 3 to 14 skills, with higher probability of role-relevant skills
        role_skills = role_skill_map[role]
        n_role_skills = np.random.randint(2, min(len(role_skills) + 1, 9))
        chosen_role_skills = list(np.random.choice(role_skills, size=n_role_skills, replace=False))
        other_skills = [s for s in all_skill_pool if s not in chosen_role_skills]
        n_other_skills = np.random.randint(1, 6)
        chosen_other_skills = list(np.random.choice(other_skills, size=n_other_skills, replace=False))
        candidate_skills = chosen_role_skills + chosen_other_skills
        random.shuffle(candidate_skills)

        # Professional accomplishments
        certs = int(np.random.poisson(lam=1.4))
        certs = min(certs, 6)
        projects = int(np.random.poisson(lam=exp * 0.7 + 1.8))
        projects = min(projects, 12)
        pubs = int(np.random.poisson(lam=0.4 if edu_level in ["Masters", "PhD"] else 0.05))
        pubs = min(pubs, 5)

        # Career dynamics
        gap = int(np.random.choice([0, 0, 0, 0, 3, 6, 9, 12, 18, 24], p=[0.55, 0.15, 0.10, 0.05, 0.05, 0.04, 0.02, 0.02, 0.01, 0.01]))
        notice = int(np.random.choice([0, 15, 30, 60, 90], p=[0.15, 0.15, 0.40, 0.20, 0.10]))
        
        # Expected salary in LPA correlated with experience
        base_salary = 3.5 + exp * 2.2 + (2.0 if tier == "Tier 1" else 0.0) + (2.5 if role in ["Data Scientist", "ML Engineer"] else 0.0)
        expected_salary = round(float(np.clip(np.random.normal(base_salary, 2.5), 3.5, 55.0)), 1)

        # Previous job titles
        if exp < 1.0:
            prev_title = "Fresher / Intern"
        elif exp < 3.5:
            prev_title = f"Associate {role}"
        elif exp < 7.0:
            prev_title = role
        else:
            prev_title = f"Senior {role}"

        # Resume text generation
        top_skills_str = ", ".join(candidate_skills[:4])
        tmpl = templates[i % len(templates)]
        resume_text = tmpl.format(
            role=role, exp=exp, edu=edu_level, degree=deg_field,
            tier=tier, top_skills=top_skills_str, projects=projects,
            certs=certs, gpa=gpa, company_type=comp_type, pubs=pubs, gap=gap
        )

        # TARGET GENERATION LOGIC:
        # 1. Merit Components:
        # Skill match ratio
        overlap = len(set(candidate_skills) & set(role_skills))
        skill_match_ratio = overlap / len(role_skills)  # 0.0 to 1.0

        # Experience score
        exp_score = min(exp / 6.0, 1.5)  # diminishing return after 6 yrs

        # Education score
        edu_score_map = {"High School": -0.8, "Diploma": -0.4, "Bachelors": 0.4, "Masters": 0.9, "PhD": 1.3}
        edu_score = edu_score_map[edu_level]

        # GPA score
        gpa_score = (gpa - 7.5) * 0.4

        # Projects and Certs
        proj_score = min(projects, 8) * 0.12
        cert_score = min(certs, 4) * 0.10
        pub_score = pubs * 0.15

        # Merit logit
        merit_logit = (
            3.0 * skill_match_ratio +
            0.6 * exp_score +
            0.5 * edu_score +
            0.3 * gpa_score +
            0.4 * proj_score +
            0.2 * cert_score +
            0.2 * pub_score
        )

        # 2. Intentionally Injected MILD Historical Bias:
        # - Small penalty for employment gaps > 6 months
        gap_penalty = -0.35 if gap >= 6 else ( -0.65 if gap >= 12 else 0.0 )
        
        # - Small advantage for Tier 1 college
        tier_bias = 0.35 if tier == "Tier 1" else (-0.15 if tier == "Tier 3" else 0.0)
        
        # - Slight gender skew (synthetic demonstration)
        gender_bias = 0.22 if gender == "Male" else (-0.15 if gender == "Female" else -0.20)
        
        # - Slight age skew (penalty for candidates over 45)
        age_bias = -0.40 if age > 45 else (0.10 if (26 <= age <= 36) else 0.0)

        # Overall latent score + logistic noise
        noise = np.random.logistic(loc=0.0, scale=0.7)
        total_logit = (
            merit_logit +
            gap_penalty +
            tier_bias +
            gender_bias +
            age_bias +
            noise - 3.45  # calibration intercept for ~27% target rate
        )

        prob = 1.0 / (1.0 + np.exp(-total_logit))
        shortlisted = 1 if prob >= 0.50 else 0

        candidates.append({
            "candidate_id": cand_id,
            "name": name,
            "gender": gender,
            "age": age,
            "location": location,
            "education_level": edu_level,
            "degree_field": deg_field,
            "college_tier": tier,
            "gpa_or_percentage": gpa,
            "years_experience": exp,
            "previous_job_titles": prev_title,
            "current_company_type": comp_type,
            "skills": ", ".join(candidate_skills),
            "certifications_count": certs,
            "projects_count": projects,
            "publications_count": pubs,
            "employment_gap_months": gap,
            "notice_period_days": notice,
            "expected_salary_lpa": expected_salary,
            "job_role_applied": role,
            "resume_text": resume_text,
            "shortlisted": shortlisted
        })

    df = pd.DataFrame(candidates)

    # 3. Inject Realistic Messiness for Data Cleaning:
    # A) Missing values (3% - 4.5% in some columns)
    mask_gpa = np.random.rand(n_samples) < 0.04
    df.loc[mask_gpa, "gpa_or_percentage"] = np.nan

    mask_tier = np.random.rand(n_samples) < 0.035
    df.loc[mask_tier, "college_tier"] = np.nan

    mask_company = np.random.rand(n_samples) < 0.03
    df.loc[mask_company, "current_company_type"] = np.nan

    mask_salary = np.random.rand(n_samples) < 0.04
    df.loc[mask_salary, "expected_salary_lpa"] = np.nan

    # B) Messy casing and synonyms in degree_field
    def mess_degree(deg):
        r = np.random.rand()
        if r < 0.12:
            return f"B.Tech in {deg}" if "CS" in deg or "IT" in deg else f"BTech {deg}"
        elif r < 0.20:
            return deg.lower()
        elif r < 0.25:
            return f"Bachelor of Science - {deg}"
        return deg

    df["degree_field"] = df["degree_field"].apply(mess_degree)

    # C) Messy location variants
    city_variants = {
        "Bengaluru": ["bengaluru", "Bangalore", "BANGALORE", "Bengaluru"],
        "Mumbai": ["Mumbai", "Bombay", "mumbai"],
        "Delhi NCR": ["Delhi NCR", "gurgaon", "Gurugram", "Noida", "delhi"],
        "Pune": ["pune", "Pune", "PUNE"]
    }
    def mess_city(loc):
        if loc in city_variants and np.random.rand() < 0.35:
            return np.random.choice(city_variants[loc])
        return loc

    df["location"] = df["location"].apply(mess_city)

    # D) Messy skills casing / spacing
    def mess_skills(skills_str):
        if np.random.rand() < 0.25:
            tokens = [s.strip() for s in skills_str.split(",")]
            messy_tokens = []
            for t in tokens:
                p = np.random.rand()
                if p < 0.15:
                    messy_tokens.append(f" {t.lower()} ")
                elif p < 0.25:
                    messy_tokens.append(f"  {t}  ")
                else:
                    messy_tokens.append(t)
            return ",".join(messy_tokens)
        return skills_str

    df["skills"] = df["skills"].apply(mess_skills)

    # E) Years of experience formatted as string in ~5% of rows ("5 yrs", "3.5 Years", etc.)
    def mess_exp(val):
        if np.random.rand() < 0.05:
            variant = np.random.choice(["yrs", "years", "Years", "Yrs"])
            return f"{val} {variant}"
        return val

    # We store as object column to simulate raw CSV input
    df["years_experience"] = df["years_experience"].apply(mess_exp)

    # F) Inject a few realistic outliers (e.g. 15 rows)
    outlier_indices = np.random.choice(n_samples, size=15, replace=False)
    for idx in outlier_indices[:5]:
        df.loc[idx, "gpa_or_percentage"] = 11.2  # impossible GPA > 10
    for idx in outlier_indices[5:10]:
        df.loc[idx, "expected_salary_lpa"] = 320.0  # extreme salary outlier
    for idx in outlier_indices[10:]:
        df.loc[idx, "years_experience"] = "48 yrs"  # age vs exp outlier

    # G) Inject a few duplicate rows (e.g. 25 duplicates)
    dup_indices = np.random.choice(n_samples, size=25, replace=False)
    dup_rows = df.iloc[dup_indices].copy()
    # Replace last 25 rows with duplicates to keep exactly 15,000 entries
    df.iloc[-25:] = dup_rows.values

    return df

def main():
    print("=" * 60)
    print("Generating Synthetic Resume Dataset (15,000 Candidates)...")
    print("=" * 60)

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_csv = os.path.join(raw_dir, "resumes_15000.csv")

    df = generate_candidates(n_samples=15000, seed=42)
    df.to_csv(out_csv, index=False)

    print(f"\n[SUCCESS] Dataset generated and saved to: {os.path.abspath(out_csv)}")
    print(f"Shape: {df.shape}")
    print("\nTarget Class Distribution:")
    balance = df['shortlisted'].value_counts(normalize=True) * 100
    for cls, pct in balance.items():
        print(f"  Class {cls}: {df['shortlisted'].value_counts()[cls]:,d} ({pct:.2f}%)")

    print("\nMissing Values Count (Top Columns):")
    nulls = df.isnull().sum()
    for col, count in nulls[nulls > 0].items():
        print(f"  {col}: {count} ({count / len(df) * 100:.2f}%)")

    print("\nGender Distribution:")
    print(df['gender'].value_counts(normalize=True) * 100)

    print("\nJob Roles Applied:")
    print(df['job_role_applied'].value_counts())

if __name__ == "__main__":
    main()
