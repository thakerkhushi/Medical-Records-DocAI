"""Utility helpers."""
import re
from pathlib import Path


PATIENT_ID_PATTERN = re.compile(r"([A-Z]{3}-\d{4}-PA-\d{7})")
PAGE_PATTERN = re.compile(r"_page_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def extract_patient_id(file_name: str) -> str:
    """Extract the patient ID from a dataset filename."""
    match = PATIENT_ID_PATTERN.search(file_name)
    return match.group(1) if match else "UNKNOWN"


def extract_page_number(file_name: str) -> int:
    """Extract the page number from a dataset filename."""
    match = PAGE_PATTERN.search(file_name)
    return int(match.group(1)) if match else 0


def detect_document_type(file_name: str, text: str) -> str:
    """Infer document type from file name and OCR text."""
    haystack = f"{file_name} {text[:500]}".lower()
    if "lab" in haystack or "hematology" in haystack or "pathology" in haystack:
        return "lab_report"
    if "discharge" in haystack or "summary" in haystack:
        return "discharge_summary"
    if "bill" in haystack or "invoice" in haystack:
        return "billing"
    if "preauth" in haystack or "query" in haystack or "claim" in haystack:
        return "insurance"
    if "prescription" in haystack or "medicine" in haystack:
        return "prescription"
    return "general_medical_record"


def clean_text(text: str) -> str:
    """Normalize OCR output into searchable text."""
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_page_id(path: Path, patient_id: str, page_number: int) -> str:
    """Create a deterministic page ID."""
    stem = path.stem[:60].replace(" ", "_")
    return f"{patient_id}::{stem}::{page_number}"
