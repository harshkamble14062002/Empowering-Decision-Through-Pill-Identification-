import os
"""Filesystem and upload configuration for the production web application."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_TEMPLATE = PROJECT_ROOT / "frontend" / "index.html"
DATABASE_FILE = PROJECT_ROOT / "database" / "runtime" / "medicine_details_verified.csv"
RAG_INDEX_FILE = PROJECT_ROOT / "database" / "runtime" / "medicine_rag_sbert_index.joblib"
MULTICLASS_MODEL_FILE = PROJECT_ROOT / "multiclass_models" / "best_multiclass.pt"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "live"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ENABLE_MULTICLASS = os.environ.get("ENABLE_MULTICLASS", "true").lower() in {"1", "true", "yes"}
