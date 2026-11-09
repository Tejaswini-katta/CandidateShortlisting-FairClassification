"""
Structured Resume Parser
========================
Extracts structured candidate information from raw text, PDF, or DOCX resumes.
Extracts name, education level, degree field, years of experience, skills,
certifications, projects, publications, and employment gap hints.
Returns a dictionary compatible with the model's feature schema.
"""

import os
import re
from typing import Dict, Any, List, Optional

# Synonyms and Canonical Skills Dictionary
SKILL_SYNONYMS = {
    "python": "Python",
    "py": "Python",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Deep Learning",
    "dl": "Deep Learning",
    "sql": "SQL",
    "mysql": "SQL",
    "postgresql": "SQL",
    "postgres": "SQL",
    "pl/sql": "SQL",
    "statistics": "Statistics",
    "statistical analysis": "Statistics",
    "pandas": "Pandas",
    "numpy": "Pandas",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "data visualization": "Data Visualization",
    "visualization": "Data Visualization",
    "matplotlib": "Data Visualization",
    "seaborn": "Data Visualization",
    "aws": "AWS",
    "amazon web services": "AWS",
    "docker": "Docker",
    "containers": "Docker",
    "mlops": "MLOps",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "git": "Git",
    "github": "Git",
    "version control": "Git",
    "excel": "Excel",
    "ms excel": "Excel",
    "advanced excel": "Excel",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "tableau": "Tableau",
    "reporting": "Reporting",
    "communication": "Communication",
    "verbal communication": "Communication",
    "java": "Java",
    "rest apis": "REST APIs",
    "rest api": "REST APIs",
    "restful": "REST APIs",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "data structures": "Data Structures",
    "dsa": "Data Structures",
    "system design": "System Design",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "jenkins": "CI/CD",
    "agile": "Agile",
    "scrum": "Agile",
    "jira": "Jira",
    "requirements gathering": "Requirements Gathering",
    "problem solving": "Problem Solving",
    "analytical thinking": "Problem Solving",
    "process mapping": "Process Mapping",
    "react": "React",
    "react.js": "React",
    "reactjs": "React",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "node": "Node.js",
    "c++": "C++",
    "cpp": "C++",
    "linux": "Linux",
    "unix": "Linux",
    "mongodb": "MongoDB",
    "nosql": "MongoDB",
    "spark": "Spark",
    "pyspark": "Spark",
    "leadership": "Leadership",
    "team lead": "Leadership"
}

def extract_text_from_pdf(file_path_or_bytes) -> str:
    """Extract text from PDF file path or file-like buffer using pdfplumber with PyPDF fallback."""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(file_path_or_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback to PyPDF2 or pypdf
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file_path_or_bytes)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception:
        pass

    return text

def extract_text_from_docx(file_path_or_bytes) -> str:
    """Extract text from DOCX file using python-docx."""
    import docx
    doc = docx.Document(file_path_or_bytes)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                full_text.append(cell.text)
    return "\n".join(full_text)

def extract_name(text: str) -> str:
    """Extract candidate name from header or top lines of resume text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    ignore_words = {"resume", "curriculum vitae", "cv", "profile", "summary", "contact", "experience", "education", "skills"}
    for line in lines[:5]:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', line).strip()
        words = cleaned.split()
        if 2 <= len(words) <= 4 and cleaned.lower() not in ignore_words:
            if not any(w.lower() in ignore_words for w in words):
                return cleaned.title()
    return "Candidate Name"

def extract_education_level(text: str) -> str:
    """Extract highest education level: PhD, Masters, Bachelors, Diploma, High School."""
    lower = text.lower()
    if re.search(r'\b(ph\.?d|doctorate|doctoral)\b', lower):
        return "PhD"
    if re.search(r'\b(master|m\.?tech|m\.?s|mca|mba|m\.?sc|m\.?e)\b', lower):
        return "Masters"
    if re.search(r'\b(bachelor|b\.?tech|btech|b\.?e|b\.?sc|bca|b\.?com|bba|undergraduate)\b', lower):
        return "Bachelors"
    if re.search(r'\b(diploma|polytechnic)\b', lower):
        return "Diploma"
    if re.search(r'\b(high school|12th|hsc|senior secondary)\b', lower):
        return "High School"
    return "Bachelors"

def extract_degree_field(text: str) -> str:
    """Extract degree discipline or field of study."""
    lower = text.lower()
    if re.search(r'\b(computer science|cse|cs)\b', lower):
        return "Computer Science"
    if re.search(r'\b(information technology|it)\b', lower):
        return "Information Technology"
    if re.search(r'\b(data science|artificial intelligence|ai & ds)\b', lower):
        return "Data Science"
    if re.search(r'\b(electronics|ece|electrical|eee)\b', lower):
        return "Electronics"
    if re.search(r'\b(mechanical|mech|civil)\b', lower):
        return "Mechanical"
    if re.search(r'\b(commerce|finance|accounting|economics)\b', lower):
        return "Commerce"
    if re.search(r'\b(arts|humanities|business|management)\b', lower):
        return "Arts"
    return "Computer Science"

def extract_college_tier(text: str) -> str:
    """Detect college tier heuristics from text."""
    lower = text.lower()
    tier_1_keywords = ["iit", "nit", "bits pilani", "iisc", "iiit", "iim", "tier 1", "tier-1", "top tier"]
    tier_2_keywords = ["state university", "anna university", "vtu", "vit", "manipal", "srm", "tier 2", "tier-2"]
    if any(k in lower for k in tier_1_keywords):
        return "Tier 1"
    if any(k in lower for k in tier_2_keywords):
        return "Tier 2"
    return "Tier 3"

def extract_years_experience(text: str) -> float:
    """Extract total years of experience using regular expressions."""
    lower = text.lower()
    
    # Check explicit statement e.g. "6.5 years of experience", "4+ yrs experience"
    exp_matches = re.findall(r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?', lower)
    if exp_matches:
        vals = [float(m) for m in exp_matches if float(m) <= 45.0]
        if vals:
            return round(max(vals), 1)

    # Check date ranges e.g. "2018 - 2024", "2019 to Present"
    current_year = 2026
    year_ranges = re.findall(r'(20\d\d)\s*(?:-|–|to)\s*(20\d\d|present|current)', lower)
    if year_ranges:
        total_yrs = 0.0
        for start, end in year_ranges:
            s_yr = int(start)
            e_yr = current_year if end in ["present", "current"] else int(end)
            if e_yr >= s_yr and s_yr >= 1990:
                total_yrs += (e_yr - s_yr)
        if 0.0 < total_yrs <= 40.0:
            return round(total_yrs, 1)

    return 3.0  # sensible default if unstated

def extract_skills(text: str) -> List[str]:
    """Match skills against the synonyms dictionary and return unique list of canonical skills."""
    lower = text.lower()
    found_skills = set()
    
    # Sort keys by length descending to match multi-word phrases first
    sorted_keywords = sorted(SKILL_SYNONYMS.keys(), key=lambda x: len(x), reverse=True)
    
    for kw in sorted_keywords:
        # Regex word boundary check (handle special characters like c++, ci/cd)
        escaped_kw = re.escape(kw)
        pattern = rf'(?<![a-zA-Z0-9]){escaped_kw}(?![a-zA-Z0-9])'
        if re.search(pattern, lower):
            found_skills.add(SKILL_SYNONYMS[kw])

    if not found_skills:
        found_skills = {"Communication", "Problem Solving"}
        
    return sorted(list(found_skills))

def extract_certifications_count(text: str) -> int:
    """Count mentioned certifications or credentials."""
    lower = text.lower()
    cert_keywords = ["certified", "certification", "credential", "coursera", "aws certified", "deeplearning.ai", "license"]
    count = 0
    for kw in cert_keywords:
        matches = re.findall(rf'\b{re.escape(kw)}\b', lower)
        count += len(matches)
    
    # Check explicit section or number
    num_match = re.search(r'(\d+)\s*(?:recognized\s+)?certifications?', lower)
    if num_match:
        return min(int(num_match.group(1)), 8)
    return min(count, 6)

def extract_projects_count(text: str) -> int:
    """Estimate number of projects completed."""
    lower = text.lower()
    num_match = re.search(r'(\d+)\s*(?:impactful|production|enterprise|industry)?\s*projects?', lower)
    if num_match:
        return min(int(num_match.group(1)), 15)
    
    # Count bullet items or occurrences under a Projects header
    project_matches = re.findall(r'\bproject[s]?\b', lower)
    if project_matches:
        return min(len(project_matches), 8)
    return 3

def extract_publications_count(text: str) -> int:
    """Count publications or papers."""
    lower = text.lower()
    num_match = re.search(r'(\d+)\s*(?:technical\s+)?publications?', lower)
    if num_match:
        return min(int(num_match.group(1)), 6)
    if re.search(r'\b(publication|ieee|arxiv|springer|conference paper)\b', lower):
        return 1
    return 0

def extract_employment_gap(text: str) -> int:
    """Detect employment gap in months."""
    lower = text.lower()
    gap_match = re.search(r'(\d+)\s*(?:months?|mths?)\s*(?:of\s+)?(?:employment\s+)?gap', lower)
    if gap_match:
        return int(gap_match.group(1))
    if "career break" in lower or "sabbatical" in lower:
        return 12
    return 0

def extract_gpa(text: str) -> float:
    """Extract GPA or percentage and normalize to 10.0 scale."""
    match = re.search(r'(?:gpa|cgpa|score)[:\s]*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if val > 10.0 and val <= 100.0:
            return round(val / 10.0, 2)
        if 0.0 < val <= 10.0:
            return round(val, 2)
    return 7.5

def extract_company_type(text: str) -> str:
    """Determine previous company type."""
    lower = text.lower()
    if "product" in lower or "faang" in lower:
        return "Product"
    if "mnc" in lower:
        return "MNC"
    if "startup" in lower:
        return "Startup"
    if "service" in lower or "consulting" in lower or "tcs" in lower or "infosys" in lower:
        return "Service"
    return "MNC"

def parse_resume(file_path_or_text: Any, default_role: str = "Data Scientist") -> Dict[str, Any]:
    """
    Parse a resume from raw text, PDF path, or DOCX path, returning a clean dictionary
    matching the candidate schema.
    """
    if isinstance(file_path_or_text, str) and os.path.exists(file_path_or_text):
        ext = os.path.splitext(file_path_or_text)[1].lower()
        if ext == ".pdf":
            raw_text = extract_text_from_pdf(file_path_or_text)
        elif ext in [".docx", ".doc"]:
            raw_text = extract_text_from_docx(file_path_or_text)
        else:
            with open(file_path_or_text, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
    elif hasattr(file_path_or_text, "read"):
        # File-like object (e.g. Streamlit uploaded file)
        filename = getattr(file_path_or_text, "name", "resume.txt").lower()
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path_or_text)
        elif filename.endswith(".docx"):
            raw_text = extract_text_from_docx(file_path_or_text)
        else:
            raw_text = file_path_or_text.read().decode("utf-8", errors="ignore")
    else:
        raw_text = str(file_path_or_text)

    skills_list = extract_skills(raw_text)
    exp = extract_years_experience(raw_text)
    edu = extract_education_level(raw_text)
    degree = extract_degree_field(raw_text)
    tier = extract_college_tier(raw_text)
    gpa = extract_gpa(raw_text)
    certs = extract_certifications_count(raw_text)
    projects = extract_projects_count(raw_text)
    pubs = extract_publications_count(raw_text)
    gap = extract_employment_gap(raw_text)
    comp_type = extract_company_type(raw_text)
    name = extract_name(raw_text)

    # Heuristic for age and salary
    min_age = 22 if edu in ["Bachelors", "High School", "Diploma"] else (24 if edu == "Masters" else 27)
    est_age = int(min_age + exp)
    est_salary = round(3.5 + exp * 2.2, 1)

    return {
        "candidate_id": "PARSED-001",
        "name": name,
        "gender": "Unknown",  # Excluded from training features
        "age": est_age,        # Excluded from training features
        "location": "Bengaluru",
        "education_level": edu,
        "degree_field": degree,
        "college_tier": tier,
        "gpa_or_percentage": gpa,
        "years_experience": exp,
        "previous_job_titles": f"Specialist ({default_role})",
        "current_company_type": comp_type,
        "skills": ", ".join(skills_list),
        "certifications_count": certs,
        "projects_count": projects,
        "publications_count": pubs,
        "employment_gap_months": gap,
        "notice_period_days": 30,
        "expected_salary_lpa": est_salary,
        "job_role_applied": default_role,
        "resume_text": raw_text.strip()
    }

