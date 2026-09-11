"""Check a running container, including OCR without external network access."""
import io
import time

from PIL import Image
import requests

from app.rag import resolve_confirmed_medicine


def main():
    base = "http://127.0.0.1:8000"
    for attempt in range(30):
        try:
            response = requests.get(f"{base}/healthz", timeout=3)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 29:
                raise
            time.sleep(2)

    assert response.json()["openrouter_configured"] is False
    assert requests.get(f"{base}/ocr-csv", timeout=5).status_code == 200
    assert requests.get(f"{base}/static/app.js", timeout=5).status_code == 200
    assert resolve_confirmed_medicine("aa 5 tablet")["status"] == "success"

    image = io.BytesIO()
    Image.new("RGB", (320, 160), "white").save(image, format="PNG")
    response = requests.post(
        f"{base}/api/analyze",
        files={"image": ("blank.png", image.getvalue(), "image/png")},
        data={"question": "What is this medicine?"},
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    assert result["status"] == "identification_uncertain", result["status"]
    assert result["medicine"] is None
    assert result["annotated_image"]
    print("Container passed: web routes, stored index, pretrained OCR and uncertain results.")


if __name__ == "__main__":
    main()
