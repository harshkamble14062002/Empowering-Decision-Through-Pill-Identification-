"""Web interface for medicine identification and bilingual questions."""

import asyncio
import base64
from functools import lru_cache
import mimetypes
from pathlib import Path
import tempfile
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.llm import OpenRouterGenerator
from app.ocr import run_ocr_csv_pipeline
from app.rag import DEFAULT_DATABASE, DEFAULT_INDEX, answer_kannada, detect_language

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PIPELINE_LOCK = threading.Lock()

app = FastAPI(title="Pill Detection", version="1.0")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@lru_cache(maxsize=1)
def get_generator():
    try:
        return OpenRouterGenerator()
    except ValueError:
        return None


def image_data_url(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def analyze(upload, filename, question):
    suffix = Path(filename or "upload.jpg").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError("Use a JPG, PNG, WEBP, or BMP medicine-cover image.")
    if not question.strip():
        raise ValueError("Enter a Kannada or English medicine question.")
    question = question.strip()

    with tempfile.TemporaryDirectory(prefix="pill-") as directory:
        temporary = Path(directory)
        image_path = temporary / f"upload{suffix}"
        image_path.write_bytes(upload)
        # EasyOCR shares a reader; serialize requests on this process.
        with PIPELINE_LOCK:
            identification = run_ocr_csv_pipeline(
                image_path,
                database_path=DEFAULT_DATABASE,
                output_dir=temporary / "output",
                quiet=True,
            )
            if identification["status"] == "success":
                medicine_name = identification["final_result"]["medicine_name"]
                chatbot = answer_kannada(
                    medicine_name, question, DEFAULT_INDEX, get_generator()
                )
                status = chatbot["status"]
            else:
                status = "identification_uncertain"
                chatbot = {
                    "answer": (
                        "ಔಷಧಿಯನ್ನು ಖಚಿತವಾಗಿ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ಆದ್ದರಿಂದ ಔಷಧಿ ಮಾಹಿತಿಯನ್ನು ನೀಡಲಾಗುವುದಿಲ್ಲ."
                        if detect_language(question) == "kn"
                        else "The medicine could not be identified confidently, so its information cannot be provided."
                    )
                }

        stages = identification.get("stages", {})
        annotated = stages.get("detection", {}).get("annotated_path")
        return {
            "status": status,
            "answer": chatbot.get("answer"),
            "llm_output": chatbot.get("llm_output"),
            "medicine": identification.get("final_result"),
            "evidence": chatbot.get("evidence"),
            "generation": chatbot.get("generation"),
            "timings": identification.get("timings"),
            "uncertainty_reasons": identification.get("uncertainty_reasons", []),
            "identification_method": identification.get("match_strategy"),
            "ocr_regions": stages.get("ocr_csv", {}).get("ocr_regions", []),
            "candidates": identification.get("candidates", []),
            "annotated_image": (
                image_data_url(annotated)
                if annotated and Path(annotated).is_file()
                else None
            ),
        }


@app.get("/", response_class=HTMLResponse)
@app.get("/ocr-csv", response_class=HTMLResponse)
def home():
    return (FRONTEND / "index.html").read_text(encoding="utf-8")


@app.get("/healthz")
def health():
    configured = get_generator() is not None
    return {
        "status": "ok" if configured else "configuration_required",
        "openrouter_configured": configured,
    }


@app.post("/api/analyze")
async def analyze_endpoint(image: UploadFile = File(...), question: str = Form(...)):
    upload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(upload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be 10 MB or smaller.")
    try:
        return await asyncio.to_thread(analyze, upload, image.filename, question)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            503, "Could not process this image. Please try again."
        ) from exc
