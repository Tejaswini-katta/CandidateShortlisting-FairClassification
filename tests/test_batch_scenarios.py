import os
import io
import sys
import pytest
import pandas as pd
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.resume_parser import parse_resume
from src.cleaning import clean_dataset
from src.score import score_candidates

class MockUploadedFile(io.BytesIO):
    """Mocks a Streamlit UploadedFile object."""
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name
        self.size = len(content)

def process_resume_file(uploaded_file, default_role: str = "Data Scientist") -> dict:
    return parse_resume(uploaded_file, default_role=default_role)

def run_batch_simulation(uploaded_batch_files, model, preprocessor):
    """Simulates the exact Batch Candidate Ranking logic in app/streamlit_app.py."""
    processed_records = []
    failed_files = []
    warn_few_fields = []

    for idx, file_obj in enumerate(uploaded_batch_files):
        fname = getattr(file_obj, "name", f"file_{idx+1}")
        fsize = getattr(file_obj, "size", 0)

        # 1. Size limit (10 MB)
        if fsize > 10 * 1024 * 1024:
            failed_files.append((fname, "File size exceeds 10 MB limit."))
            continue

        ext = os.path.splitext(fname)[1].lower()

        # 2. CSV file flow
        if ext == ".csv":
            try:
                file_obj.seek(0)
                df_csv = pd.read_csv(file_obj)
                if df_csv.empty:
                    failed_files.append((fname, "CSV file is empty."))
                else:
                    df_csv["source_file"] = fname
                    if "candidate_id" not in df_csv.columns:
                        base_id = os.path.splitext(fname)[0]
                        df_csv["candidate_id"] = [f"{base_id}_{r+1}" for r in range(len(df_csv))]
                    if "name" not in df_csv.columns:
                        df_csv["name"] = df_csv["candidate_id"]
                    for r in df_csv.to_dict(orient="records"):
                        processed_records.append(r)
            except Exception as e:
                failed_files.append((fname, f"Corrupted or invalid CSV format ({str(e)})."))

        # 3. PDF, DOCX, TXT individual candidate flow
        elif ext in [".pdf", ".docx", ".doc", ".txt"]:
            try:
                if fsize == 0:
                    failed_files.append((fname, "File is empty (0 bytes)."))
                    continue

                file_obj.seek(0)
                cand_dict = process_resume_file(file_obj, default_role="Data Scientist")
                raw_text = cand_dict.get("resume_text", "")
                if not raw_text or not raw_text.strip():
                    failed_files.append((fname, "No extractable text found (scanned image or empty document)."))
                    continue

                base_filename = os.path.splitext(fname)[0]
                if not cand_dict.get("name") or cand_dict.get("name") in ["Candidate Name", "Candidate", "Unknown"]:
                    cand_dict["name"] = base_filename.replace("_", " ").title()
                if not cand_dict.get("candidate_id") or cand_dict.get("candidate_id") == "PARSED-001":
                    cand_dict["candidate_id"] = base_filename

                cand_dict["source_file"] = fname
                cand_dict["shortlisted"] = 0

                # Check for sparse extraction
                extracted_count = 0
                if cand_dict.get("skills") and cand_dict.get("skills") != "Communication, Problem Solving":
                    extracted_count += 1
                if cand_dict.get("years_experience", 0.0) != 3.0:
                    extracted_count += 1
                if cand_dict.get("education_level") in ["Masters", "PhD", "Diploma", "High School"]:
                    extracted_count += 1
                if cand_dict.get("degree_field") and cand_dict.get("degree_field") != "Computer Science":
                    extracted_count += 1
                if cand_dict.get("projects_count", 0) > 0 and cand_dict.get("projects_count", 0) != 3:
                    extracted_count += 1
                if cand_dict.get("certifications_count", 0) > 0:
                    extracted_count += 1
                if cand_dict.get("employment_gap_months", 0) > 0:
                    extracted_count += 1
                if cand_dict.get("gpa_or_percentage", 0.0) != 7.5:
                    extracted_count += 1

                if extracted_count < 3:
                    warn_few_fields.append(fname)

                processed_records.append(cand_dict)

            except Exception as e:
                failed_files.append((fname, f"Corrupted or unreadable document ({str(e)})."))
        else:
            failed_files.append((fname, f"Unsupported file type '{ext}'."))

    summary_text = f"Processed {len(processed_records)} files successfully, {len(failed_files)} failed."
    
    if not processed_records:
        return {
            "summary": summary_text,
            "processed_count": 0,
            "failed_count": len(failed_files),
            "failed_files": failed_files,
            "warn_few_fields": warn_few_fields,
            "scored_df": None,
            "csv_download_size": 0
        }

    raw_batch_df = pd.DataFrame(processed_records)
    if "shortlisted" not in raw_batch_df.columns:
        raw_batch_df["shortlisted"] = 0
    else:
        raw_batch_df["shortlisted"] = raw_batch_df["shortlisted"].fillna(0).astype(int)
    cleaned_batch, _ = clean_dataset(raw_batch_df)

    scored_batch = score_candidates(cleaned_batch, model=model, preprocessor=preprocessor)

    # Simulate filtering
    filtered_df = scored_batch.copy()
    filtered_df = filtered_df[filtered_df['suitability_score'] >= 0]
    
    # Simulate CSV download preparation
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')

    return {
        "summary": summary_text,
        "processed_count": len(processed_records),
        "failed_count": len(failed_files),
        "failed_files": failed_files,
        "warn_few_fields": warn_few_fields,
        "scored_df": scored_batch,
        "filtered_df": filtered_df,
        "csv_download_size": len(csv_bytes)
    }

@pytest.fixture(scope="module")
def model_and_prep():
    model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.joblib"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.joblib"))
    return model, preprocessor

def test_scenario_a_three_pdfs_only(model_and_prep):
    """Scenario A: 3 PDFs only."""
    model, preprocessor = model_and_prep
    sample_dir = os.path.join(BASE_DIR, "sample_resumes")
    files = []
    for pdf_name in ["01_strong_data_scientist.pdf", "02_strong_ml_engineer.pdf", "03_strong_data_analyst.pdf"]:
        path = os.path.join(sample_dir, pdf_name)
        with open(path, "rb") as f:
            files.append(MockUploadedFile(f.read(), pdf_name))

    res = run_batch_simulation(files, model, preprocessor)
    assert res["processed_count"] == 3
    assert res["failed_count"] == 0
    assert "Processed 3 files successfully, 0 failed." == res["summary"]
    scored = res["scored_df"]
    assert len(scored) == 3
    assert "source_file" in scored.columns
    assert set(scored["source_file"]) == {"01_strong_data_scientist.pdf", "02_strong_ml_engineer.pdf", "03_strong_data_analyst.pdf"}
    assert "suitability_score" in scored.columns
    assert "match_tier" in scored.columns
    assert res["csv_download_size"] > 0
    print("\nScenario A PASSED:", res["summary"])

def test_scenario_b_one_csv_only(model_and_prep):
    """Scenario B: 1 CSV only."""
    model, preprocessor = model_and_prep
    csv_path = os.path.join(BASE_DIR, "data", "raw", "resumes_15000.csv")
    df_sample = pd.read_csv(csv_path, nrows=5)
    csv_bytes = df_sample.to_csv(index=False).encode('utf-8')
    files = [MockUploadedFile(csv_bytes, "batch_candidates.csv")]

    res = run_batch_simulation(files, model, preprocessor)
    assert res["processed_count"] == 5
    assert res["failed_count"] == 0
    assert "Processed 5 files successfully, 0 failed." == res["summary"]
    scored = res["scored_df"]
    assert len(scored) == 5
    assert "source_file" in scored.columns
    assert all(scored["source_file"] == "batch_candidates.csv")
    assert all(scored["suitability_score"] >= 0)
    assert res["csv_download_size"] > 0
    print("\nScenario B PASSED:", res["summary"])

def test_scenario_c_mixed_pdfs_docx_csv(model_and_prep):
    """Scenario C: 2 PDFs + 1 DOCX + 1 CSV together."""
    model, preprocessor = model_and_prep
    sample_dir = os.path.join(BASE_DIR, "sample_resumes")
    files = []

    # 2 PDFs
    for pdf_name in ["01_strong_data_scientist.pdf", "02_strong_ml_engineer.pdf"]:
        with open(os.path.join(sample_dir, pdf_name), "rb") as f:
            files.append(MockUploadedFile(f.read(), pdf_name))

    # 1 DOCX
    docx_name = "03_strong_data_analyst.docx"
    with open(os.path.join(sample_dir, docx_name), "rb") as f:
        files.append(MockUploadedFile(f.read(), docx_name))

    # 1 CSV (2 candidates)
    csv_path = os.path.join(BASE_DIR, "data", "raw", "resumes_15000.csv")
    df_sample = pd.read_csv(csv_path, nrows=2)
    csv_bytes = df_sample.to_csv(index=False).encode('utf-8')
    files.append(MockUploadedFile(csv_bytes, "mixed_batch.csv"))

    res = run_batch_simulation(files, model, preprocessor)
    # Total candidates: 2 (from PDFs) + 1 (from DOCX) + 2 (from CSV) = 5
    assert res["processed_count"] == 5
    assert res["failed_count"] == 0
    assert "Processed 5 files successfully, 0 failed." == res["summary"]
    scored = res["scored_df"]
    assert len(scored) == 5
    assert "source_file" in scored.columns
    assert set(scored["source_file"]) == {"01_strong_data_scientist.pdf", "02_strong_ml_engineer.pdf", "03_strong_data_analyst.docx", "mixed_batch.csv"}
    assert res["csv_download_size"] > 0
    print("\nScenario C PASSED:", res["summary"])

def test_scenario_d_corrupted_file_mixed_with_valid(model_and_prep):
    """Scenario D: 1 corrupted file mixed with valid ones."""
    model, preprocessor = model_and_prep
    sample_dir = os.path.join(BASE_DIR, "sample_resumes")
    files = []

    # Valid PDF
    with open(os.path.join(sample_dir, "01_strong_data_scientist.pdf"), "rb") as f:
        files.append(MockUploadedFile(f.read(), "valid_candidate.pdf"))

    # Corrupted PDF (random garbage bytes)
    corrupted_bytes = b"%PDF-1.4\nInvalid binary junk here that cannot be parsed as a PDF 1234567890"
    files.append(MockUploadedFile(corrupted_bytes, "corrupted_resume.pdf"))

    # Empty TXT file
    files.append(MockUploadedFile(b"", "empty_resume.txt"))

    # Valid TXT resume
    with open(os.path.join(sample_dir, "02_strong_ml_engineer.txt"), "rb") as f:
        files.append(MockUploadedFile(f.read(), "valid_ml_engineer.txt"))

    res = run_batch_simulation(files, model, preprocessor)
    # 2 valid files succeeded, 2 invalid files failed
    assert res["processed_count"] == 2
    assert res["failed_count"] == 2
    assert "Processed 2 files successfully, 2 failed." == res["summary"]
    failed_names = [f[0] for f in res["failed_files"]]
    assert "corrupted_resume.pdf" in failed_names
    assert "empty_resume.txt" in failed_names
    # Check that failed files have clear human-readable reasons (no raw traceback)
    for fname, reason in res["failed_files"]:
        assert len(reason) > 5
        assert "Traceback" not in reason

    scored = res["scored_df"]
    assert len(scored) == 2
    assert set(scored["source_file"]) == {"valid_candidate.pdf", "valid_ml_engineer.txt"}
    print("\nScenario D PASSED:", res["summary"], "Failed files:", res["failed_files"])
