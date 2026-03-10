"""Persistence and in-memory application state."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.config import settings
from app.extractor import ClinicalExtractor
from app.ocr import OCRService
from app.retriever import Retriever
from app.schemas import PageRecord, PatientSummary
from app.utils import (
    detect_document_type,
    extract_page_number,
    extract_patient_id,
    make_page_id,
)


class DataStore:
    """Loads OCR pages, patient summaries, and the search index."""

    def __init__(self) -> None:
        self.ocr_service = OCRService()
        self.extractor = ClinicalExtractor()
        self.pages: list[PageRecord] = []
        self.patient_summaries: dict[str, PatientSummary] = {}
        self.retriever = Retriever([])

    def ingest_directory(self, dataset_dir: str | Path) -> dict[str, int]:
        """Process a dataset directory of image pages."""
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        page_records: list[PageRecord] = []
        patient_to_texts: dict[str, list[str]] = defaultdict(list)

        image_paths = sorted(
            [
                path
                for path in dataset_path.rglob("*")
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )

        for image_path in image_paths:
            page_record = self._load_or_process_page(image_path)
            page_records.append(page_record)
            patient_to_texts[page_record.patient_id].append(page_record.text)

        summaries = {
            patient_id: self.extractor.build_summary(patient_id, texts)
            for patient_id, texts in patient_to_texts.items()
        }

        self.pages = page_records
        self.patient_summaries = summaries
        self.retriever = Retriever(page_records)
        return {
            "page_count": len(page_records),
            "patient_count": len(summaries),
        }

    def list_patients(self) -> list[dict[str, str | int]]:
        """Return lightweight patient cards."""
        return [
            {
                "patient_id": patient_id,
                "page_count": summary.page_count,
                "diagnosis_count": len(summary.diagnoses),
                "medication_count": len(summary.medications),
            }
            for patient_id, summary in sorted(self.patient_summaries.items())
        ]

    def get_patient_summary(self, patient_id: str) -> PatientSummary:
        """Return a patient summary or raise KeyError."""
        return self.patient_summaries[patient_id]

    def _load_or_process_page(self, image_path: Path) -> PageRecord:
        """Load cached OCR result or create it."""
        cache_path = settings.cache_dir / f"{image_path.name}.json"
        if cache_path.exists():
            return PageRecord.model_validate_json(cache_path.read_text(encoding="utf-8"))

        patient_id = extract_patient_id(image_path.name)
        page_number = extract_page_number(image_path.name)
        text = self.ocr_service.extract_text(image_path)
        document_type = detect_document_type(image_path.name, text)
        page_record = PageRecord(
            page_id=make_page_id(image_path, patient_id, page_number),
            patient_id=patient_id,
            source_file=image_path.name,
            page_number=page_number,
            text=text,
            document_type=document_type,
        )
        cache_path.write_text(page_record.model_dump_json(indent=2), encoding="utf-8")
        return page_record


store = DataStore()
