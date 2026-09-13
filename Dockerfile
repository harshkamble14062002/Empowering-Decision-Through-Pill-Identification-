FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EASYOCR_MODULE_PATH=/opt/easyocr \
    ENABLE_MULTICLASS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN mkdir -p /opt/easyocr \
    && python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" \
    && chmod -R a+rX /opt/easyocr

COPY backend/ backend/
COPY database/runtime/ database/runtime/
COPY extract_text.py kannada_rag.py main_chatbot.py ocr_csv_classifier.py openrouter_llm.py streamlit_app.py ./

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8501') + '/_stcore/health', timeout=3)"]

CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
