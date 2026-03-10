"""Search and grounded question answering over OCR records."""
from __future__ import annotations

import math
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.config import settings
from app.schemas import AskResponse, PageRecord, SearchResult


class Retriever:
    """Simple TF-IDF retriever with optional LLM answering."""

    def __init__(self, pages: list[PageRecord]) -> None:
        self.pages = pages
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([page.text for page in pages]) if pages else None
        self.client = (
            OpenAI(api_key=settings.openai_api_key)
            if settings.openai_api_key and OpenAI is not None
            else None
        )

    def search(self, query: str, patient_id: str | None = None, top_k: int = 5) -> list[SearchResult]:
        """Return the most relevant pages for a query."""
        if not self.pages or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.matrix).flatten()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        results = []
        for index, score in ranked:
            if math.isclose(score, 0.0):
                continue
            page = self.pages[index]
            if patient_id and page.patient_id != patient_id:
                continue
            snippet = _build_snippet(page.text, query)
            results.append(
                SearchResult(
                    score=float(score),
                    patient_id=page.patient_id,
                    page_id=page.page_id,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    snippet=snippet,
                    document_type=page.document_type,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def answer(self, question: str, patient_id: str) -> AskResponse:
        """Answer a patient-specific question using retrieved context."""
        search_results = self.search(
            query=question,
            patient_id=patient_id,
            top_k=settings.top_k_search_results,
        )
        citations = [result.model_dump() for result in search_results]
        if not search_results:
            return AskResponse(
                answer="I could not find relevant evidence in the uploaded records.",
                citations=[],
            )

        context = "\n\n".join(
            [
                (
                    f"[Citation {index + 1}] file={item.source_file} "
                    f"page={item.page_number}\n{item.snippet}"
                )
                for index, item in enumerate(search_results)
            ]
        )

        if self.client:
            answer = self._answer_with_llm(question=question, patient_id=patient_id, context=context)
        else:
            answer = (
                "LLM is not configured. Here is the strongest grounded evidence I found:\n\n"
                f"{context}"
            )
        return AskResponse(answer=answer, citations=citations)

    def _answer_with_llm(self, question: str, patient_id: str, context: str) -> str:
        prompt = f"""
Answer the question using only the cited evidence below.
If the answer is unclear, say so.
Mention citation numbers inline, like [Citation 1].
Patient ID: {patient_id}
Question: {question}

Evidence:
{context}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer questions only from supplied medical record evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or "No answer returned."
        except Exception:
            return "The LLM call failed, but the citations below still contain the grounded evidence."



def _build_snippet(text: str, query: str, width: int = 350) -> str:
    lowered_text = text.lower()
    lowered_query = query.lower().strip()
    index = lowered_text.find(lowered_query)
    if index == -1:
        return text[:width]
    start = max(index - 80, 0)
    end = min(index + width, len(text))
    return text[start:end].replace("\n", " ")
