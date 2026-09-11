# Pill Detection

A medicine-cover reader with questions and answers in English and Kannada. Upload a package photo, choose one of 20 Kannada questions or type your own, and view the matched medicine and its details.

## How it works

EasyOCR reads the packaging text. The matcher groups nearby words, ignores packaging details such as batch numbers, and compares brand, composition and manufacturer text with the local medicine catalogue. Strength and formulation mismatches reduce the match score.

When a match is accepted, the app retrieves the corresponding record and sends that row with the question to OpenRouter (`openrouter/free`). The answer follows the question's language. Uncertain matches stop before answer generation. If OpenRouter is unavailable, the app displays a local catalogue-based answer; descriptive fields in that fallback may remain in English.

This branch contains the full-image OCR application. Earlier YOLO experiments and training datasets are outside this branch.

## Run locally

Use Python 3.11. The default dependencies use CPU inference.

```sh
git clone --branch showcase --single-branch https://github.com/harshkamble14062002/Empowering-Decision-Through-Pill-Identification-.git pill-detection
cd pill-detection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`, then start the server:

```sh
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [localhost:8000](http://localhost:8000). The `/ocr-csv` URL serves the same page. Without an API key, identification and local answers still work. The first OCR request downloads EasyOCR's pretrained weights; later requests reuse them.

## Files

```text
app/
  __init__.py        Python package
  main.py            FastAPI routes and request flow
  ocr.py             Text extraction and medicine matching
  rag.py             Record retrieval and bilingual answer handling
  llm.py             OpenRouter client
frontend/
  index.html         Upload form and Kannada questions
  app.js             Form submission and results
  styles.css         Page styles
data/
  medicines.csv      Medicine catalogue
  medicine_index.joblib  Indexed rows and saved embeddings
tests/
  test_api.py        Requests, upload limits and uncertain results
  test_matching.py   OCR matching rules
  test_answers.py    Retrieved evidence and language behavior
  test_llm.py        OpenRouter requests and error handling
  container_smoke.py Running container and offline OCR checks
.github/workflows/tests.yml  Automated tests
.env.example         Configuration template
.gitignore           Files excluded from Git
.dockerignore        Files excluded from container builds
Dockerfile           Container setup
compose.yaml         One-command Docker startup
requirements.txt     Runtime dependencies
requirements-dev.txt Test dependencies
README.md            Setup and project notes
```

## Data and results

The catalogue contains 7,975 records with medicine name, composition, manufacturer, uses, side effects and review percentages. It is the existing project dataset; these fields have not been independently medically verified. Review percentages are values from that dataset, not measurements made by this application.

The supplied index stores the same rows and 384-dimensional embeddings from `paraphrase-multilingual-MiniLM-L12-v2`. The current image workflow looks up the confirmed medicine by name in those indexed rows; it does not run a separate vector search for each question. No embedding model or LLM runs locally during that lookup. EasyOCR's pretrained detection and recognition models run locally.

An earlier local evaluation on 50 internally reviewed images produced 36 correct, 0 wrong and 14 uncertain identifications. This is a small development sample, not an independent accuracy benchmark. Identification can fail on blurry images, missing strengths, unfamiliar brands or incomplete catalogue entries. The app is an educational prototype and does not provide treatment or dosing advice.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

The automated tests mock OCR or OpenRouter where needed and do not make external LLM calls. They check matching, retrieval, language selection, uncertain results and the web API.

## Docker

Install Docker Desktop on Windows/macOS, or Docker Engine with the Compose plugin on Linux. Use Linux containers. Extract the shared project ZIP and open a terminal in the folder containing `compose.yaml`.

```sh
docker compose up --build -d
```

Open [localhost:8000](http://localhost:8000). No local Python installation or GPU is needed. The first build needs internet access to download Python dependencies and OCR weights. Compose runs a Linux AMD64 image with a 2 GiB memory limit; Docker Desktop on Apple Silicon uses AMD64 emulation and can be slower. This setup uses CPU inference.

For OpenRouter answers, copy `.env.example` to `.env`, enter your own API key, and run `docker compose up -d` again. Identification and local catalogue answers work with the key left blank. Internet access is required for OpenRouter answers. Change `APP_PORT` in `.env` if port 8000 is already occupied.

```sh
docker compose ps          # Show status
docker compose logs -f     # Read logs
docker compose down        # Stop the application
```

The API accepts JPG, PNG, WEBP and BMP uploads up to 10 MB. Requests are processed one at a time within the worker. Images and previews are temporary; no volume is required.

### Share with another person

Share the `showcase` branch ZIP, or the prepared `pill-detection-docker.zip`. The recipient extracts it and runs the Compose command above. The source package includes the medicine catalogue and saved index. API keys are not included.

If you have already built the image, you can also share it without requiring another build:

```sh
docker save -o pill-detection-image.tar pill-detection:local
```

Send that TAR together with `compose.yaml` and `.env.example`. The recipient runs:

```sh
docker load -i pill-detection-image.tar
docker compose up -d --no-build
```

Keep your personal `.env` file out of shared packages. The Docker build excludes it, and Compose passes configuration at startup. GitHub Actions builds the image and checks the running API, stored index and OCR with the container's external networking disabled.

`GET /healthz` reports whether an OpenRouter key is configured; it does not test the provider connection. `POST /api/analyze` accepts multipart fields named `image` and `question`. Uploaded images and generated previews are removed after each request. Only the question and matched catalogue row are sent to OpenRouter.
