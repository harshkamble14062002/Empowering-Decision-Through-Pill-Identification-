import csv

from kannada_rag import answer_kannada, build_index, detect_intent, detect_language, retrieve


def sample_index(tmp_path):
    database = tmp_path / "medicine.csv"
    fields = ["Medicine Name", "Composition", "Uses", "Side_effects", "Manufacturer",
              "Excellent Review %", "Average Review %", "Poor Review %"]
    with database.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"Medicine Name": "azithral 500 tablet", "Composition": "Azithromycin (500mg)",
             "Uses": "Treatment of bacterial infections", "Side_effects": "Nausea Diarrhea",
             "Manufacturer": "Alembic Pharmaceuticals Ltd", "Excellent Review %": "80",
             "Average Review %": "15", "Poor Review %": "5"},
            {"Medicine Name": "aciloc 150 tablet", "Composition": "Ranitidine (150mg)",
             "Uses": "Treatment of acid reflux", "Side_effects": "Headache Diarrhea",
             "Manufacturer": "Cadila Pharmaceuticals Ltd", "Excellent Review %": "70",
             "Average Review %": "20", "Poor Review %": "10"},
        ])
    index = tmp_path / "index.joblib"
    build_index(database, index)
    return index


def test_kannada_intents():
    assert detect_intent("ಇದರ ಉಪಯೋಗ ಏನು?") == "uses"
    assert detect_intent("ಅಡ್ಡ ಪರಿಣಾಮಗಳು ಯಾವುವು?") == "side_effects"
    assert detect_intent("ತಯಾರಕ ಕಂಪನಿ ಯಾವುದು?") == "manufacturer"
    assert detect_intent("ಈ ಔಷಧಿಯ ಹೆಸರು ಏನು?") == "medicine_name"
    assert detect_intent("ಈ ಔಷಧಿಯ ಶಕ್ತಿ ಎಷ್ಟು?") == "composition"
    assert detect_intent("ಉತ್ತಮ ವಿಮರ್ಶೆಗಳ ಶೇಕಡಾವಾರು ಎಷ್ಟು?") == "excellent_review"
    assert detect_intent("ಎಲ್ಲಾ ವಿಮರ್ಶಾ ಶೇಕಡಾವಾರುಗಳನ್ನು ತಿಳಿಸಿ") == "reviews"


def test_grounded_kannada_use_answer(tmp_path):
    index = sample_index(tmp_path)
    result = answer_kannada("azithral 500 tablet", "ಇದರ ಉಪಯೋಗ ಏನು?", index)
    assert result["status"] == "success"
    assert result["intent"] == "uses"
    assert "bacterial infections" in result["answer"]
    assert result["evidence"] == {
        "Medicine Name": "azithral 500 tablet",
        "Composition": "Azithromycin (500mg)",
        "Uses": "Treatment of bacterial infections",
        "Side_effects": "Nausea Diarrhea",
        "Manufacturer": "Alembic Pharmaceuticals Ltd",
        "Excellent Review %": "80",
        "Average Review %": "15",
        "Poor Review %": "5",
    }


def test_kannada_review_summary_uses_csv_percentages(tmp_path):
    index = sample_index(tmp_path)
    result = answer_kannada(
        "azithral 500 tablet",
        "ಈ ಔಷಧಿಯ ಎಲ್ಲಾ ವಿಮರ್ಶಾ ಶೇಕಡಾವಾರುಗಳನ್ನು ತಿಳಿಸಿ.",
        index,
    )
    assert result["intent"] == "reviews"
    assert "80%" in result["answer"]
    assert "15%" in result["answer"]
    assert "5%" in result["answer"]


def test_dosage_is_not_invented(tmp_path):
    index = sample_index(tmp_path)
    result = answer_kannada("azithral 500 tablet", "ಡೋಸ್ ಎಷ್ಟು ತೆಗೆದುಕೊಳ್ಳಬೇಕು?", index)
    assert result["status"] == "success"
    assert result["intent"] == "dosage"
    assert result["evidence"]["Medicine Name"] == "azithral 500 tablet"
    assert "ಲಭ್ಯವಿಲ್ಲ" in result["answer"]


def test_unknown_medicine_is_not_answered(tmp_path):
    index = sample_index(tmp_path)
    result = answer_kannada("completely unknown product", "ಇದರ ಉಪಯೋಗ ಏನು?", index)
    assert result["status"] == "no_match"
    assert result["evidence"] is None


def test_retrieval_uses_index(tmp_path):
    index = sample_index(tmp_path)
    results = retrieve("azithromycin bacterial infection", index, limit=1)
    assert results[0]["medicine"]["Medicine Name"] == "azithral 500 tablet"
    assert results[0]["score"] > 0


def test_generator_receives_complete_matched_csv_row(tmp_path):
    class Generator:
        backend = "test_generator"
        last_model = "test/model"

        def answer(self, medicine_name, question, evidence):
            assert medicine_name == "azithral 500 tablet"
            assert evidence == {
                "Medicine Name": "azithral 500 tablet",
                "Composition": "Azithromycin (500mg)",
                "Uses": "Treatment of bacterial infections",
                "Side_effects": "Nausea Diarrhea",
                "Manufacturer": "Alembic Pharmaceuticals Ltd",
                "Excellent Review %": "80",
                "Average Review %": "15",
                "Poor Review %": "5",
            }
            return "ಬ್ಯಾಕ್ಟೀರಿಯಾ ಸೋಂಕಿನ ಚಿಕಿತ್ಸೆಗೆ ಬಳಸಲಾಗುತ್ತದೆ."

    index = sample_index(tmp_path)
    result = answer_kannada(
        "azithral 500 tablet", "ಇದರ ಉಪಯೋಗ ಏನು?", index, Generator()
    )
    assert result["generation"]["generated"] is True
    assert result["generation"]["backend"] == "test_generator"
    assert result["generation"]["model"] == "test/model"
    assert result["answer"] == "ಬ್ಯಾಕ್ಟೀರಿಯಾ ಸೋಂಕಿನ ಚಿಕಿತ್ಸೆಗೆ ಬಳಸಲಾಗುತ್ತದೆ."
    assert result["llm_output"] == result["answer"]


def test_generator_receives_full_row_for_missing_dosage(tmp_path):
    class Generator:
        backend = "test_generator"
        last_model = "test/model"

        def answer(self, medicine_name, question, evidence):
            assert evidence["Medicine Name"] == "azithral 500 tablet"
            assert "Dosage" not in evidence
            return "ಈ ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಡೋಸ್ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ."

    index = sample_index(tmp_path)
    result = answer_kannada(
        "azithral 500 tablet", "ಡೋಸ್ ಎಷ್ಟು ತೆಗೆದುಕೊಳ್ಳಬೇಕು?", index, Generator()
    )
    assert result["generation"]["generated"] is True
    assert result["llm_output"] == result["answer"]


def test_question_language_detection():
    assert detect_language("What is it used for?") == "en"
    assert detect_language("ಇದರ ಉಪಯೋಗ ಏನು?") == "kn"


def test_english_question_gets_english_fallback(tmp_path):
    index = sample_index(tmp_path)
    result = answer_kannada("azithral 500 tablet", "What is this medicine used for?", index)
    assert result["status"] == "success"
    assert result["language"] == "en"
    assert result["answer"].startswith("Recorded uses:")
    assert result["evidence"]["Medicine Name"] == "azithral 500 tablet"
    assert result["evidence"]["Uses"] == "Treatment of bacterial infections"


def test_unknown_medicine_error_matches_question_language(tmp_path):
    index = sample_index(tmp_path)
    result = answer_kannada("unknown product", "What is it used for?", index)
    assert result["language"] == "en"
    assert result["answer"].startswith("The medicine could not")
