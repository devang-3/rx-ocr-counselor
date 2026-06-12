from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deployment.serialization import prescription_from_dict
from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.counseling.engine import PatientCounselorEngine
from prescription_pipeline.pipeline import PrescriptionPipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = REPO_ROOT / "output"

app = FastAPI(title="Prescription Pipeline API", version="1.2.0")

DEFAULT_COUNSELOR = "qwen2.5-0.5b-instruct"
_extract_pipeline: PrescriptionPipeline | None = None
_counselor_engine: PatientCounselorEngine | None = None


class ProcessRequest(BaseModel):
    input_path: str = Field(description="Image file or directory path visible to the server")
    output_dir: str = "output"
    ner_backend: str = "auto"
    patient_query: str = ""
    need_counselor: bool = False
    medication_index: int | None = None
    counselor_model: str | None = None
    counselor_temperature: float = 0.1
    save_overlays: bool = True
    enable_drug_matching: bool = True


class CounselRequest(BaseModel):
    prescription: dict[str, Any]
    patient_query: str
    need_counselor: bool = False
    medication_index: int | None = None
    counselor_model: str | None = None


def _build_config(req: ProcessRequest) -> PipelineConfig:
    config = PipelineConfig(
        output_dir=Path(req.output_dir),
        ner_backend=req.ner_backend,
        save_overlays=req.save_overlays,
        enable_drug_matching=req.enable_drug_matching,
        patient_query=req.patient_query,
        counsel_medication_index=req.medication_index,
        counselor_temperature=req.counselor_temperature,
        enable_counseling=req.need_counselor and bool(req.patient_query.strip()),
    )
    if req.counselor_model:
        config.counselor_model = req.counselor_model
    return config


def _get_extract_pipeline() -> PrescriptionPipeline:
    global _extract_pipeline
    if _extract_pipeline is None:
        _extract_pipeline = PrescriptionPipeline(
            PipelineConfig(enable_counseling=False, output_dir=OUTPUT_DIR)
        )
    return _extract_pipeline


def _get_counselor_engine(model_name: str | None = None) -> PatientCounselorEngine:
    global _counselor_engine
    model = model_name or DEFAULT_COUNSELOR
    if _counselor_engine is None or _counselor_engine.config.counselor_model != model:
        _counselor_engine = PatientCounselorEngine(
            PipelineConfig(
                enable_counseling=True,
                counselor_model=model,
                counselor_max_tokens=128,
                counselor_temperature=0.1,
                use_compiled_counselor=False,
            )
        )
    return _counselor_engine


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
def process(req: ProcessRequest) -> dict[str, Any]:
    input_path = Path(req.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Input path not found: {input_path}")

    pipeline = PrescriptionPipeline(_build_config(req))
    payload = pipeline.process_path(input_path)
    return payload


@app.post("/process-file")
async def process_file(
    file: UploadFile = File(...),
    patient_query: str = Form(default=""),
    need_counselor: bool = Form(default=False),
    ner_backend: str = Form(default="auto"),
    counselor_model: str | None = Form(default=None),
) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)

    req = ProcessRequest(
        input_path=str(tmp_path),
        output_dir="output",
        ner_backend=ner_backend,
        patient_query=patient_query,
        need_counselor=need_counselor,
        counselor_model=counselor_model,
    )
    try:
        if need_counselor and patient_query.strip():
            pipeline = PrescriptionPipeline(_build_config(req))
            result = pipeline.process_image(tmp_path).to_dict()
        else:
            result = _get_extract_pipeline().process_image(tmp_path).to_dict()
        return {"image_count": 1, "prescriptions": [result]}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/counsel")
def counsel(req: CounselRequest) -> dict[str, Any]:
    question = req.patient_query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="patient_query is required")

    if not req.need_counselor:
        return {
            "patient_query": question,
            "patient_response": (
                "Counselor is OFF. Medication facts above come from the database only. "
                "Turn on Need Counselor to chat with the LLM."
            ),
            "counseling_backend": "disabled",
        }

    try:
        rx = prescription_from_dict(req.prescription)
        counseling = _get_counselor_engine(req.counselor_model).counsel_prescription(
            rx,
            question,
            medication_index=req.medication_index,
        )
        return counseling.__dict__
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Counselor model missing. Run: python pipeline_models/download_models.py {DEFAULT_COUNSELOR}. {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


overlay_dir = OUTPUT_DIR / "overlays"
overlay_dir.mkdir(parents=True, exist_ok=True)
app.mount("/overlays", StaticFiles(directory=str(overlay_dir)), name="overlays")
