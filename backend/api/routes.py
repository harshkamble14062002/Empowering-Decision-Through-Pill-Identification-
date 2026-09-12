"""Local web interface for medicine-image identification and Kannada answers."""
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

from backend.core.config import (
    ALLOWED_IMAGE_SUFFIXES, DATABASE_FILE, ENABLE_MULTICLASS, FRONTEND_TEMPLATE, MAX_UPLOAD_BYTES,
    MULTICLASS_MODEL_FILE, RAG_INDEX_FILE,
)
from backend.services.chatbot import run as run_chatbot
from backend.services.llm import OpenRouterGenerator


PIPELINE_LOCK = threading.Lock()
app = FastAPI(title="Kannada Medicine Assistant", version="1.0")
app.mount("/static", StaticFiles(directory=FRONTEND_TEMPLATE.parent / "static"), name="static")


@lru_cache(maxsize=1)
def generator():
    return OpenRouterGenerator()


def image_data_url(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def analyze(upload, filename, question, identification_backend="ocr_csv"):
    suffix = Path(filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("Use a JPG, PNG, WEBP, or BMP medicine-cover image.")
    if not question.strip():
        raise ValueError("Enter a Kannada or English medicine question.")
    with tempfile.TemporaryDirectory(prefix="pill-live-") as directory:
        root = Path(directory)
        image_path = root / f"upload{suffix}"
        image_path.write_bytes(upload)
        with PIPELINE_LOCK:
            result = run_chatbot(
                image_path, question.strip(), database=DATABASE_FILE, index=RAG_INDEX_FILE,
                output=root / "output", quiet=True, generator=generator(),
                identification_backend=identification_backend, model=MULTICLASS_MODEL_FILE,
            )
        identification = result.get("identification", {})
        final = identification.get("final_result")
        annotated = identification.get("stages", {}).get("detection", {}).get("annotated_path")
        chatbot = result.get("chatbot", {})
        response = {
            "status": result.get("status"),
            "answer": chatbot.get("answer"),
            "llm_output": chatbot.get("llm_output"),
            "medicine": final,
            "evidence": chatbot.get("evidence"),
            "generation": chatbot.get("generation"),
            "timings": identification.get("timings"),
            "uncertainty_reasons": identification.get("uncertainty_reasons", []),
            "requested_backend": identification_backend,
            "identification_method": identification.get("match_strategy"),
            "ocr_regions": identification.get("stages", {}).get("ocr_csv", {}).get("ocr_regions", []),
            "candidates": [
                {"medicine_name": item.get("medicine_name"), "score": item.get("score")}
                for item in identification.get("candidates", [])
            ],
            "annotated_image": image_data_url(annotated) if annotated and Path(annotated).is_file() else None,
        }
        return response


@app.get("/healthz")
def health():
    try:
        configured = bool(generator().api_key)
    except ValueError:
        configured = False
    return {"status": "ok" if configured else "configuration_required",
            "openrouter_configured": configured}


async def analyze_request(image, question, identification_backend):
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be 10 MB or smaller.")
    try:
        return await asyncio.to_thread(
            analyze, data, image.filename or "upload.jpg", question, identification_backend
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, f"Processing failed: {exc}") from exc


@app.post("/api/analyze")
async def analyze_endpoint(image: UploadFile = File(...), question: str = Form(...)):
    return await analyze_request(image, question, "ocr_csv")


@app.post("/api/analyze/multiclass")
async def analyze_multiclass_endpoint(image: UploadFile = File(...), question: str = Form(...)):
    if not ENABLE_MULTICLASS:
        raise HTTPException(404, "Multiclass comparison is disabled on this deployment.")
    return await analyze_request(image, question, "multiclass")


def render_page(title, method, api_endpoint, alternate_href=None, alternate_label=None):
    template = FRONTEND_TEMPLATE.read_text()
    return (template.replace("__TITLE__", title)
            .replace("__METHOD__", method)
            .replace("__API_ENDPOINT__", api_endpoint)
            .replace("__ALT_HREF__", alternate_href or "")
            .replace("__ALT_LABEL__", alternate_label or ""))

@app.get("/", response_class=HTMLResponse)
@app.get("/ocr-csv", response_class=HTMLResponse)
def ocr_csv_page():
    return render_page(
        "Medicine Assistant",
        "Full-image EasyOCR + CSV/RapidFuzz",
        "/api/analyze", "/multiclass", "Open multiclass YOLO UI",
    )


@app.get("/multiclass", response_class=HTMLResponse)
def multiclass_page():
    if not ENABLE_MULTICLASS:
        raise HTTPException(404, "Multiclass comparison is disabled on this deployment.")
    return render_page(
        "Medicine Assistant — Multiclass YOLO",
        "YOLO brand/composition/manufacturer crops + EasyOCR",
        "/api/analyze/multiclass", "/ocr-csv", "Open full-image OCR UI",
    )

