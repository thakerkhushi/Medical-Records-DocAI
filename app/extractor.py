"""Clinical extraction logic using rules plus optional LLM refinement."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from app.config import settings
from app.schemas import PatientSummary


class ClinicalExtractor:
    """Extract clinically relevant fields from OCR text."""

    def __init__(self) -> None:
        self.client = (
            OpenAI(api_key=settings.openai_api_key)
            if settings.openai_api_key and OpenAI is not None
            else None
        )

    def build_summary(self, patient_id: str, texts: list[str]) -> PatientSummary:
        """Generate a structured patient summary."""
        combined_text = "\n\n".join(texts)[: settings.max_chars_per_patient]
        summary = PatientSummary(
            patient_id=patient_id,
            page_count=len(texts),
            diagnoses=self._extract_diagnoses(combined_text),
            medications=self._extract_medications(combined_text),
            lab_results=self._extract_lab_results(combined_text),
            allergies=self._extract_allergies(combined_text),
            vitals=self._extract_vitals(combined_text),
            other_clinical_notes=self._extract_notes(combined_text),
        )
        if self.client:
            refined = self._refine_with_llm(patient_id=patient_id, text=combined_text)
            if refined:
                summary = self._merge_llm_summary(summary, refined)
        return summary

    @staticmethod
    def _extract_diagnoses(text: str) -> list[str]:
        patterns = [
            r"diagnosis\s*[:\-]\s*(.+)",
            r"final diagnosis\s*[:\-]\s*(.+)",
            r"provisional diagnosis\s*[:\-]\s*(.+)",
            r"impression\s*[:\-]\s*(.+)",
        ]
        results = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(1).split("\n")[0].strip(" .")
                if 4 <= len(value) <= 160:
                    results.append(value)
        return _deduplicate(results)

    @staticmethod
    def _extract_medications(text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines()]
        medication_lines = []
        for line in lines:
            lowered = line.lower()
            if any(keyword in lowered for keyword in ["tab ", "tablet", "capsule", "inj ", "syrup", "mg", "ml"]):
                if 3 <= len(line) <= 120:
                    medication_lines.append(line)
        return _deduplicate(medication_lines[:20])

    @staticmethod
    def _extract_lab_results(text: str) -> list[dict[str, Any]]:
        pattern = re.compile(
            r"(?P<name>[A-Za-z][A-Za-z .()/%-]{2,40})\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z/%]+)?",
        )
        likely_labs = []
        lab_keywords = {
            "hemoglobin",
            "r.b.c",
            "w. b. c",
            "platelet",
            "creatinine",
            "sodium",
            "potassium",
            "glucose",
            "bilirubin",
            "urea",
            "hb",
        }
        for match in pattern.finditer(text):
            name = match.group("name").strip()
            normalized = name.lower()
            if any(keyword in normalized for keyword in lab_keywords):
                likely_labs.append(
                    {
                        "test_name": name,
                        "value": match.group("value"),
                        "unit": match.group("unit") or "",
                    }
                )
        return _deduplicate_dicts(likely_labs)[:20]

    @staticmethod
    def _extract_allergies(text: str) -> list[str]:
        matches = re.findall(r"allerg(?:y|ies)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
        if not matches and "no known allergy" in text.lower():
            return ["No known allergy"]
        return _deduplicate([item.split("\n")[0].strip(" .") for item in matches])

    @staticmethod
    def _extract_vitals(text: str) -> list[dict[str, Any]]:
        vital_patterns = {
            "blood_pressure": r"(?:bp|blood pressure)\s*[:\-]?\s*(\d{2,3}/\d{2,3})",
            "pulse": r"(?:pulse|pr)\s*[:\-]?\s*(\d{2,3})",
            "temperature": r"(?:temp|temperature)\s*[:\-]?\s*(\d{2,3}(?:\.\d+)?)",
            "spo2": r"(?:spo2|oxygen saturation)\s*[:\-]?\s*(\d{2,3})",
        }
        vitals = []
        for name, pattern in vital_patterns.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                vitals.append({"name": name, "value": match.group(1)})
        return _deduplicate_dicts(vitals)

    @staticmethod
    def _extract_notes(text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        notes = []
        for line in lines:
            lowered = line.lower()
            if any(keyword in lowered for keyword in ["admitted", "discharged", "complaint", "history", "advice"]):
                if 10 <= len(line) <= 180:
                    notes.append(line)
        return _deduplicate(notes[:20])

    def _refine_with_llm(self, patient_id: str, text: str) -> dict[str, Any] | None:
        """Use an LLM to convert noisy OCR into structured grounded output."""
        prompt = f"""
You are extracting clinical facts from OCR text. Only use the text provided.
Return strict JSON with these keys:
patient_id, diagnoses, medications, lab_results, allergies, vitals,
other_clinical_notes, llm_summary.
- lab_results must be a list of objects with test_name, value, unit.
- vitals must be a list of objects with name and value.
- Do not invent facts.
- Use short bullet-style strings for lists.
- Patient ID must stay exactly: {patient_id}

OCR TEXT:
{text}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful medical document information extractor.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return None

    @staticmethod
    def _merge_llm_summary(summary: PatientSummary, llm_data: dict[str, Any]) -> PatientSummary:
        """Merge rule-based and LLM-based outputs conservatively."""
        merged = defaultdict(list)
        for field_name in [
            "diagnoses",
            "medications",
            "allergies",
            "other_clinical_notes",
        ]:
            merged[field_name].extend(getattr(summary, field_name, []))
            merged[field_name].extend(llm_data.get(field_name, []))

        lab_results = _deduplicate_dicts(summary.lab_results + llm_data.get("lab_results", []))
        vitals = _deduplicate_dicts(summary.vitals + llm_data.get("vitals", []))

        return PatientSummary(
            patient_id=summary.patient_id,
            page_count=summary.page_count,
            diagnoses=_deduplicate(merged["diagnoses"]),
            medications=_deduplicate(merged["medications"]),
            lab_results=lab_results,
            allergies=_deduplicate(merged["allergies"]),
            vitals=vitals,
            other_clinical_notes=_deduplicate(merged["other_clinical_notes"]),
            llm_summary=llm_data.get("llm_summary"),
        )


def _deduplicate(values: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            cleaned.append(normalized)
    return cleaned



def _deduplicate_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique_items = []
    for item in items:
        serialized = json.dumps(item, sort_keys=True)
        if serialized not in seen:
            seen.add(serialized)
            unique_items.append(item)
    return unique_items
