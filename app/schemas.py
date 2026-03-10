"""Pydantic schemas for API requests and responses."""
from typing import Any

from pydantic import BaseModel, Field


class PageRecord(BaseModel):
    """Represents one OCR-processed page."""

    page_id: str
    patient_id: str
    source_file: str
    page_number: int
    text: str
    document_type: str


class SearchRequest(BaseModel):
    """Search query payload."""

    query: str = Field(min_length=2)
    patient_id: str | None = None
    top_k: int = 5


class QuestionRequest(BaseModel):
    """Question answering payload."""

    question: str = Field(min_length=3)
    patient_id: str


class IngestRequest(BaseModel):
    """Dataset ingestion request."""

    dataset_dir: str


class SearchResult(BaseModel):
    """Search result response."""

    score: float
    patient_id: str
    page_id: str
    source_file: str
    page_number: int
    snippet: str
    document_type: str


class PatientSummary(BaseModel):
    """Summarized patient output."""

    patient_id: str
    page_count: int
    diagnoses: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    lab_results: list[dict[str, Any]] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    vitals: list[dict[str, Any]] = Field(default_factory=list)
    other_clinical_notes: list[str] = Field(default_factory=list)
    llm_summary: str | None = None


class AskResponse(BaseModel):
    """Question answering result."""

    answer: str
    citations: list[dict[str, Any]]
