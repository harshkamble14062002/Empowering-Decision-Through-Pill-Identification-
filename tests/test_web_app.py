from fastapi.testclient import TestClient

from backend.main import app
from web_app import app as compatibility_app


def test_legacy_web_app_entrypoint_uses_structured_backend():
    assert compatibility_app is app


def test_home_page_has_working_post_script():
    client = TestClient(app)
    response = client.get("/")
    script = client.get("/static/app.js")
    stylesheet = client.get("/static/styles.css")
    assert response.status_code == 200
    assert script.status_code == 200
    assert stylesheet.status_code == 200
    assert 'data-api-endpoint="/api/analyze"' in response.text
    assert "fetch(form.dataset.apiEndpoint" in script.text
    assert "d.answer+'\\n\\n'" in script.text
    assert "OCR regions:" in script.text
    assert "LLM output" in script.text
    assert "Local fallback" in script.text
    assert response.text.count("<option value=") == 21
    assert "preset.onchange" in script.text
    assert "ಸ್ಥಳೀಯ ಪರ್ಯಾಯ ಉತ್ತರ" in script.text
    assert "ಚಿತ್ರದಿಂದ ಓದಿದ ಪಠ್ಯ ಪ್ರದೇಶಗಳು" in script.text


def test_both_identification_pages_have_separate_api_routes():
    client = TestClient(app)
    ocr_page = client.get("/ocr-csv")
    multiclass_page = client.get("/multiclass")
    assert ocr_page.status_code == 200
    assert "Full-image EasyOCR + CSV/RapidFuzz" in ocr_page.text
    assert 'data-api-endpoint="/api/analyze"' in ocr_page.text
    assert 'href="/multiclass"' not in ocr_page.text
    assert "Open multiclass YOLO UI" not in ocr_page.text
    assert multiclass_page.status_code == 200
    assert "YOLO brand/composition/manufacturer crops + EasyOCR" in multiclass_page.text
    assert 'data-api-endpoint="/api/analyze/multiclass"' in multiclass_page.text


def test_health_reports_openrouter_configuration():
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json()["openrouter_configured"] is True
