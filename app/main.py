"""FastAPI entrypoint for the Medical Records DocAI app."""
from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.schemas import IngestRequest, QuestionRequest, SearchRequest
from app.store import store

app = FastAPI(title="Medical Records DocAI", version="1.0.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Render the minimal frontend."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"default_data_dir": str(settings.data_dir)},
    )


@app.post("/api/ingest")
def ingest_dataset(payload: IngestRequest) -> dict[str, int | str]:
    """Ingest a local dataset directory."""
    stats = store.ingest_directory(payload.dataset_dir)
    return {"status": "ok", **stats}


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)) -> dict[str, int | str]:
    """Upload image files and ingest them."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in files:
        destination = upload_dir / uploaded_file.filename
        with destination.open("wb") as file_handle:
            shutil.copyfileobj(uploaded_file.file, file_handle)
    stats = store.ingest_directory(upload_dir)
    return {"status": "ok", **stats}


@app.get("/api/patients")
def list_patients() -> list[dict[str, str | int]]:
    """List all ingested patients."""
    return store.list_patients()


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str):
    """Return a single patient summary."""
    try:
        return store.get_patient_summary(patient_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Patient not found") from error


@app.post("/api/search")
def search_records(payload: SearchRequest):
    """Search over OCR pages."""
    return store.retriever.search(
        query=payload.query,
        patient_id=payload.patient_id,
        top_k=payload.top_k,
    )


@app.post("/api/ask")
def ask_question(payload: QuestionRequest):
    """Answer a grounded question for one patient."""
    if payload.patient_id not in store.patient_summaries:
        raise HTTPException(status_code=404, detail="Patient not found")
    return store.retriever.answer(question=payload.question, patient_id=payload.patient_id)
