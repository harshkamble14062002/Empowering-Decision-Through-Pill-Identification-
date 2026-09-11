from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


@pytest.mark.parametrize(
    "path", ["/", "/ocr-csv", "/static/app.js", "/static/styles.css"]
)
def test_browser_resources_are_served(path):
    assert client.get(path).status_code == 200


def test_health_reports_missing_key():
    with patch("app.main.get_generator", return_value=None):
        response = client.get("/healthz")
    assert response.json()["openrouter_configured"] is False


@pytest.mark.parametrize(
    "filename,question",
    [("image.txt", "What is this?"), ("image.jpg", "   ")],
)
def test_invalid_uploads_are_rejected_before_ocr(filename, question):
    with patch("app.main.run_ocr_csv_pipeline") as ocr:
        response = client.post(
            "/api/analyze",
            files={"image": (filename, b"unused", "image/jpeg")},
            data={"question": question},
        )
    assert response.status_code == 400
    ocr.assert_not_called()


def test_large_upload_is_rejected():
    with (
        patch("app.main.MAX_UPLOAD_BYTES", 8),
        patch("app.main.run_ocr_csv_pipeline") as ocr,
    ):
        response = client.post(
            "/api/analyze",
            files={"image": ("image.jpg", b"123456789", "image/jpeg")},
            data={"question": "What is this?"},
        )
    assert response.status_code == 413
    ocr.assert_not_called()


@pytest.mark.parametrize("question", ["What is this?", "ಈ ಔಷಧಿಯ ಹೆಸರು ಏನು?"])
def test_uncertain_match_does_not_reach_the_chatbot(question):
    identification = {
        "status": "uncertain",
        "final_result": None,
        "uncertainty_reasons": ["ambiguous_candidates"],
    }
    with (
        patch("app.main.run_ocr_csv_pipeline", return_value=identification),
        patch("app.main.answer_kannada") as chatbot,
    ):
        response = client.post(
            "/api/analyze",
            files={"image": ("image.jpg", b"unused", "image/jpeg")},
            data={"question": question},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "identification_uncertain"
    assert response.json()["evidence"] is None
    chatbot.assert_not_called()


def test_confirmed_match_reaches_the_chatbot():
    identification = {
        "status": "success",
        "final_result": {"medicine_name": "aa 5 tablet"},
    }
    answer = {
        "status": "success",
        "answer": "Recorded uses",
        "evidence": {"Uses": "Allergy"},
    }
    with (
        patch("app.main.run_ocr_csv_pipeline", return_value=identification),
        patch("app.main.get_generator", return_value=None),
        patch("app.main.answer_kannada", return_value=answer) as chatbot,
    ):
        response = client.post(
            "/api/analyze",
            files={"image": ("image.jpg", b"unused", "image/jpeg")},
            data={"question": "What is this used for?"},
        )
    assert response.json()["status"] == "success"
    assert response.json()["evidence"] == answer["evidence"]
    assert chatbot.call_args.args[:2] == ("aa 5 tablet", "What is this used for?")
