from unittest.mock import Mock, patch

import pytest

from app.llm import OpenRouterGenerator


def test_key_is_required(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenRouterGenerator()


def test_grounded_request_and_model_metadata():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "model": "provider/example:free",
        "choices": [{"message": {"content": '{"answer_text":"ಕನ್ನಡ ಉತ್ತರ"}'}}],
    }
    with patch("app.llm.requests.post", return_value=response) as post:
        generator = OpenRouterGenerator(api_key="test-secret")
        answer = generator.answer(
            "azithral 500 tablet",
            "ಇದರ ಉಪಯೋಗ ಏನು?",
            {"uses": "Treatment of bacterial infections"},
        )
    assert answer == "ಕನ್ನಡ ಉತ್ತರ"
    assert generator.last_model == "provider/example:free"
    request = post.call_args.kwargs
    assert request["json"]["model"] == "openrouter/free"
    assert (
        "Treatment of bacterial infections" in request["json"]["messages"][1]["content"]
    )
    assert request["headers"]["Authorization"] == "Bearer test-secret"


def test_english_question_requires_english_answer():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "model": "provider/example:free",
        "choices": [
            {
                "message": {
                    "content": '{"answer_text":"It is used for bacterial infections."}'
                }
            }
        ],
    }
    with patch("app.llm.requests.post", return_value=response) as post:
        answer = OpenRouterGenerator(api_key="test-secret").answer(
            "azithral 500 tablet",
            "What is this medicine used for?",
            {"uses": "Treatment of bacterial infections"},
        )
    assert answer == "It is used for bacterial infections."
    assert "clear English" in post.call_args.kwargs["json"]["messages"][0]["content"]


def test_free_model_400_retries_without_structured_output():
    rejected = Mock(status_code=400)
    accepted = Mock(status_code=200)
    accepted.raise_for_status.return_value = None
    accepted.json.return_value = {
        "model": "provider/free-model",
        "choices": [
            {"message": {"content": "ಈ ಔಷಧಿಯನ್ನು ಅಲರ್ಜಿ ಚಿಕಿತ್ಸೆಗೆ ಬಳಸುತ್ತಾರೆ."}}
        ],
    }
    complete_row = {
        "Medicine Name": "aa 5 tablet",
        "Composition": "Levocetirizine (5mg)",
        "Uses": "Treatment of Allergic conditions",
        "Manufacturer": "Blaze Remedies",
    }
    with patch("app.llm.requests.post", side_effect=[rejected, accepted]) as post:
        answer = OpenRouterGenerator(api_key="test-secret").answer(
            "aa 5 tablet", "ಈ ಔಷಧಿಯ ಸಂಪೂರ್ಣ ಮಾಹಿತಿ ನೀಡಿ.", complete_row
        )
    assert answer == "ಈ ಔಷಧಿಯನ್ನು ಅಲರ್ಜಿ ಚಿಕಿತ್ಸೆಗೆ ಬಳಸುತ್ತಾರೆ."
    assert post.call_count == 2
    first = post.call_args_list[0].kwargs["json"]
    second = post.call_args_list[1].kwargs["json"]
    assert "response_format" in first
    assert "response_format" not in second
    assert "reasoning" not in second
    assert "Treatment of Allergic conditions" in second["messages"][1]["content"]
