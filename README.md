# Medical Records DocAI

A grounded medical-records prototype for OCR-heavy patient documents.

## What I understood from the assignment

The core problem is **not** generic document upload. It is a noisy-document intelligence problem with four constraints:

1. The records are inconsistent, scanned, and hard to parse.
2. Patient names are unreliable or redacted.
3. The system must extract clinically useful information.
4. The system must support both **search** and **question answering** with grounding.

Because the assignment explicitly says to use the alphanumeric ID in the filename as the patient identifier, this implementation groups all pages by the filename pattern:

```text
[A-Z]{3}-\d{4}-PA-\d{7}
```

That gives stable patient grouping even when OCR quality is poor or names are missing.

## Scope

This prototype intentionally focuses on a small but complete slice:

- OCR image pages from the supplied dataset
- Group pages by patient ID from filename
- Extract structured highlights:
  - diagnoses
  - medications
  - lab results
  - allergies
  - vitals
  - other clinically relevant notes
- Search OCR text across all records or one patient
- Ask grounded questions for one patient
- Expose both a web UI and API

## Architecture

```text
Dataset images
   ↓
OCR (Tesseract + image preprocessing)
   ↓
Patient grouping from filename Patient ID
   ↓
Rule-based extraction
   ↓
Optional LLM refinement to structured JSON
   ↓
TF-IDF search index
   ↓
Grounded Q&A with citation-bearing retrieval context
   ↓
FastAPI web UI + JSON API
```

## Why this approach

### OCR
The provided dataset pages are images. That makes OCR mandatory. I used Tesseract because:

- it is local and easy to run in Codespaces
- it avoids external OCR dependency costs
- it is good enough for a short assignment

### Search
I used TF-IDF retrieval because it is simple, fast, deterministic, and cheap. For an interview assignment, that is better than pretending a full vector DB is necessary.

### Extraction
The extraction layer is hybrid:

- **baseline rule-based extraction** for predictable fields
- **optional LLM refinement** to turn noisy OCR into better structured output

This is the right tradeoff for an assignment. Pure LLM extraction without guardrails is fragile. Pure regex-only extraction is too brittle for messy medical pages.

### Grounded question answering
The QA flow retrieves the top relevant snippets first, then asks the LLM to answer **only from those snippets**. This gives:

- traceable evidence
- lower hallucination risk
- cheaper inference

## LLM usage

The assignment requires an AI/LLM component. In this project, the LLM is used in two places:

1. **Clinical extraction refinement**
   - Converts OCR text into structured JSON
   - Helps normalize diagnoses, medications, and notes

2. **Grounded question answering**
   - Answers user questions from retrieved evidence only
   - Encourages explicit citation references like `[Citation 1]`

### Model choice
Default model in `.env.example`:

```env
OPENAI_MODEL=gpt-4o-mini
```

Why this choice:

- cheap enough for an assignment
- fast enough for a UI demo
- strong structured-output capability
- works well for JSON extraction and grounded QA

You can swap the provider if needed. The code isolates LLM calls so migration is straightforward.

## Grounding strategy

This project grounds answers in three ways:

1. Retrieval happens first.
2. Only retrieved text snippets are given to the LLM.
3. The prompt explicitly forbids using outside knowledge.

## Local setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in `OPENAI_API_KEY` if you want the LLM-enabled extraction and QA.

### 4. Seed a local sample dataset

```bash
python scripts/seed_sample_data.py
```

### 5. Run the app

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```


## Assumptions

1. The dataset consists primarily of image pages, not clean machine-readable PDFs.
2. The filename patient ID is the source of truth for grouping.
3. OCR quality is noisy, so extraction will never be perfect.
4. The user asks questions for one patient at a time.
5. This is a prototype, not a production clinical safety system.

## Limitations

1. Tesseract OCR will miss text on low-quality scans.
2. TF-IDF retrieval is weaker than a good embedding index on semantic search.
3. Medication extraction is heuristic and may include false positives.
4. Lab parsing is intentionally shallow and not normalized to LOINC or ranges.
5. The UI is minimal and functional, not polished.
6. Without an API key, the LLM features fall back to non-LLM behavior.

## What I would improve next

1. Replace Tesseract-only OCR with a document OCR stack like PaddleOCR or Azure Document Intelligence.
2. Use layout-aware parsing so tables and lab panels are extracted more reliably.
3. Swap TF-IDF for hybrid retrieval: BM25 + embeddings + metadata filters.
4. Add page thumbnails and evidence highlighting in the UI.
5. Add a clinician timeline view: admission, treatment, discharge, follow-up.
6. Normalize medications, lab tests, and diagnoses to controlled vocabularies.
7. Add evaluation scripts for extraction quality and QA grounding.

## How to explain this in the interview

### 1. Problem framing
You are solving OCR-heavy patient record understanding, not general chat over files.

### 2. Why patient ID from filename matters
Names are unreliable. The filename ID is stable and required by the spec.

### 3. Why the pipeline is hybrid
Because raw OCR is messy. Rules give stability. LLM adds flexibility.

### 4. Why retrieval before QA matters
Because unguided medical QA hallucinates. Retrieval constrains the answer space.

### 5. Why the stack is intentionally simple
Because a small, working, explainable prototype beats a fake enterprise architecture.

## Suggested demo flow

1. Ingest dataset
2. Open a patient summary
3. Show extracted labs and medications
4. Search for `platelet count`
5. Ask: `What abnormal labs are mentioned?`
6. Show that the answer references retrieved evidence
