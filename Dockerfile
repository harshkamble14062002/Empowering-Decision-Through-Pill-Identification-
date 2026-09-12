FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EASYOCR_MODULE_PATH=/opt/easyocr \
    ENABLE_MULTICLASS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-production.txt .
RUN pip install --upgrade pip && pip install -r requirements-production.txt

RUN mkdir -p /opt/easyocr \
    && python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" \
    && chmod -R a+rX /opt/easyocr

COPY backend/ backend/
COPY frontend/ frontend/
COPY database/runtime/ database/runtime/
COPY main_chatbot.py kannada_rag.py openrouter_llm.py ocr_csv_classifier.py extract_text.py ./

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
