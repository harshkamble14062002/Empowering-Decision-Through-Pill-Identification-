# Kannada Medicine Assistant

A bilingual (Kannada/English) tool that identifies medicines from photos and answers questions about them. 

We built this to solve a specific problem: patients often get medicines with English packaging but only speak or read Kannada. By taking a picture of the strip, the system identifies the drug and explains its uses, side effects, and composition in their native language.

## Inputs & Outputs

**Input:**
1. An image (`JPG`, `PNG`, `WEBP`) of a medicine strip or packaging.
2. A text question (e.g., "What is this used for?" or "ಇದರ ಅಡ್ಡ ಪರಿಣಾಮಗಳೇನು?").

**Output:**
1. **Medicine Metadata:** The exact identified medicine, including Brand name, Composition, and Manufacturer.
2. **Answer:** A direct answer to the user's question, strictly grounded in the database and generated in the language of the prompt (Kannada or English).
3. **Debug Info:** Bounding box images and raw OCR text regions.

## How It Works (The Pipeline)

Instead of relying on a single black-box multimodal model, the architecture is split into deterministic, debuggable stages:

1. **OCR Text Extraction (`extract_text.py`)**  
   We use `EasyOCR` to scan the uploaded image. It returns the raw text blobs and their bounding boxes.

2. **Candidate Matching (`ocr_csv_classifier.py`)**  
   Raw OCR output from a pill bottle is messy. We pass the extracted text through `RapidFuzz` to establish a fuzzy match against our local verified dataset (`database/runtime/medicine_details_verified.csv`). The scoring algorithm handles missing characters, ignores batch numbers, and heavily penalizes dosage/strength mismatches (e.g., confusing 500mg with 650mg).

3. **Information Retrieval (`kannada_rag.py`)**  
   Once the engine commits to a medicine match, it looks up the specific row in a pre-computed local index. This retrieves the exact factual data for that medicine: uses, side effects, and ingredients.

4. **Answer Generation (`openrouter_llm.py`)**  
   We feed the retrieved factual row + the user's question into an OpenRouter LLM. The LLM's system prompt restricts it to acting strictly as a translator and summarizer. It is explicitly forbidden from inventing dosages, interactions, or medical advice not present in the provided CSV row.

## Core Project Structure

- `backend/` — FastAPI application serving the pipeline routes.
- `frontend/` — Native HTML/JS/CSS web interface.
- `database/runtime/` — The verified CSV catalogue and matching index used for local lookups.
- `streamlit_app.py` — A standalone Streamlit UI interface used for rapid cloud deployments.
- `tests/` — Pytest suite covering matching accuracy, API logic, and LLM constraints.

## Setup & Running Locally

Python 3.11+ is required.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys:**
   Copy `.env.example` to `.env` and add your OpenRouter key:
   ```env
   OPENROUTER_API_KEY="sk-or-v1-..."
   ```

3. **Start the API & Web Interface:**
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
   Open `http://localhost:8000` in your browser.

4. **(Alternative) Start the Streamlit UI:**
   If you prefer the Streamlit interface:
   ```bash
   streamlit run streamlit_app.py
   ```

## Development Team
Parshuram G P, Parvati M B, Shreyas V M, Harsha Ravindra Kamble  
**Guide:** Dr. S. Saranya Rubini (PES University)
