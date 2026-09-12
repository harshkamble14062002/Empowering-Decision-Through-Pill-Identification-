---
title: Kannada Medicine Assistant
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Kannada Medicine Assistant

A bilingual (Kannada/English) medicine identification and Q&A tool. Upload a medicine packet photo, ask a question in Kannada or English, and get an answer.

## How it works

1. **EasyOCR** reads text from the medicine packaging image
2. **RapidFuzz** matches the extracted text against 7,975 medicine records
3. **OpenRouter LLM** answers your question in Kannada or English using the matched record

## Run locally

```bash
git clone https://github.com/harshkamble14062002/Empowering-Decision-Through-Pill-Identification-.git
cd Empowering-Decision-Through-Pill-Identification-
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env
pip install -r requirements-production.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000

## Team

Parshuram G P, Parvati M B, Shreyas V M, Harsha Ravindra Kamble  
Guide: Dr. S. Saranya Rubini (PES University)  
Contact: harshkamble14062002@gmail.com
