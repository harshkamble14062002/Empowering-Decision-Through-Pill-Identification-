"""Resolve confirmed medicines and answer from the matching catalogue row."""

from functools import lru_cache
from pathlib import Path
import re

import joblib
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data" / "medicines.csv"
DEFAULT_INDEX = ROOT / "data" / "medicine_index.joblib"


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


@lru_cache(maxsize=2)
def load_index(path, modified):
    return joblib.load(path)


def resolve_confirmed_medicine(name, index=DEFAULT_INDEX):
    path = Path(index).resolve()
    payload = load_index(str(path), path.stat().st_mtime_ns)
    names = [row["Medicine Name"] for row in payload["rows"]]
    normalized = [normalize(value) for value in names]
    query = normalize(name)
    exact = [i for i, value in enumerate(normalized) if value == query]
    if len(exact) == 1:
        return {
            "status": "success",
            "match_type": "exact",
            "score": 100.0,
            "medicine": payload["rows"][exact[0]],
            "candidates": [],
        }
    ranked = process.extract(query, normalized, scorer=fuzz.ratio, limit=3)
    candidates = [
        {"medicine_name": names[index_value], "score": round(float(score), 2)}
        for _, score, index_value in ranked
    ]
    if not ranked or ranked[0][1] < 88:
        return {
            "status": "no_match",
            "medicine": None,
            "candidates": candidates,
            "reasons": ["confirmed_medicine_not_found"],
        }
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 5:
        return {
            "status": "uncertain",
            "medicine": None,
            "candidates": candidates,
            "reasons": ["ambiguous_medicine_name"],
        }
    return {
        "status": "success",
        "match_type": "fuzzy",
        "score": round(float(ranked[0][1]), 2),
        "medicine": payload["rows"][ranked[0][2]],
        "candidates": candidates,
    }


INTENT_TERMS = {
    "medicine_name": ["medicine name", "name of this medicine", "ಔಷಧಿಯ ಹೆಸರು"],
    "uses": [
        "use",
        "uses",
        "used for",
        "benefit",
        "ಉಪಯೋಗ",
        "ಬಳಕೆ",
        "ಯಾವುದಕ್ಕೆ",
        "ಏಕೆ",
        "ಯಾವ ಕಾಯಿಲೆ",
    ],
    "side_effects": [
        "side effect",
        "adverse",
        "ಅಡ್ಡ ಪರಿಣಾಮ",
        "ದುಷ್ಪರಿಣಾಮ",
        "ವಾಂತಿ",
        "ತಲೆನೋವು",
    ],
    "composition": [
        "composition",
        "ingredient",
        "contains",
        "strength",
        "ಸಂಯೋಜನೆ",
        "ಘಟಕ",
        "ಪದಾರ್ಥ",
        "ಶಕ್ತಿ",
    ],
    "manufacturer": ["manufacturer", "company", "made by", "ತಯಾರಕ", "ಕಂಪನಿ"],
    "excellent_review": ["excellent review", "good review", "ಉತ್ತಮ ವಿಮರ್ಶೆ"],
    "average_review": ["average review", "ಸರಾಸರಿ ವಿಮರ್ಶೆ"],
    "poor_review": ["poor review", "bad review", "ಕಳಪೆ ವಿಮರ್ಶೆ"],
    "reviews": [
        "all review",
        "review percentage",
        "ಎಲ್ಲಾ ವಿಮರ್ಶಾ",
        "ವಿಮರ್ಶಾ ಶೇಕಡಾವಾರು",
    ],
    "dosage": ["dose", "dosage", "how much", "ಡೋಸ್", "ಪ್ರಮಾಣ", "ಎಷ್ಟು ತೆಗೆದುಕೊಳ್ಳ"],
}


def detect_intent(question):
    lowered = (question or "").lower()
    scores = {
        intent: sum(term in lowered for term in terms)
        for intent, terms in INTENT_TERMS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "summary"


def detect_language(question):
    return "kn" if re.search(r"[\u0c80-\u0cff]", question or "") else "en"


def answer_kannada(medicine_name, question, index=DEFAULT_INDEX, generator=None):
    language = detect_language(question)
    resolution = resolve_confirmed_medicine(medicine_name, index)
    if resolution["status"] != "success":
        answer = (
            "ಔಷಧಿಯನ್ನು ಖಚಿತವಾಗಿ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಔಷಧಿಯ ಪೂರ್ಣ ಹೆಸರನ್ನು ಪರಿಶೀಲಿಸಿ."
            if language == "kn"
            else "The medicine could not be identified confidently. Please check its complete name."
        )
        return {
            "status": resolution["status"],
            "answer": answer,
            "language": language,
            "evidence": None,
            "candidates": resolution.get("candidates", []),
            "reasons": resolution.get("reasons", []),
        }
    row = resolution["medicine"]
    intent = detect_intent(question)
    values = {
        "medicine_name": row.get("Medicine Name", ""),
        "uses": row.get("Uses", ""),
        "side_effects": row.get("Side_effects", ""),
        "composition": row.get("Composition", ""),
        "manufacturer": row.get("Manufacturer", ""),
        "excellent_review": row.get("Excellent Review %", ""),
        "average_review": row.get("Average Review %", ""),
        "poor_review": row.get("Poor Review %", ""),
    }
    prefixes = {
        "kn": {
            "medicine_name": "ಈ ಔಷಧಿಯ ಹೆಸರು:",
            "uses": "ಈ ಔಷಧಿಯ ದಾಖಲಾಗಿರುವ ಬಳಕೆಗಳು:",
            "side_effects": "ದಾಖಲಾಗಿರುವ ಸಂಭವನೀಯ ಅಡ್ಡ ಪರಿಣಾಮಗಳು:",
            "composition": "ಈ ಔಷಧಿಯ ಸಂಯೋಜನೆ:",
            "manufacturer": "ಈ ಔಷಧಿಯ ತಯಾರಕರು:",
            "excellent_review": "ಉತ್ತಮ ವಿಮರ್ಶೆಗಳ ಶೇಕಡಾವಾರು:",
            "average_review": "ಸರಾಸರಿ ವಿಮರ್ಶೆಗಳ ಶೇಕಡಾವಾರು:",
            "poor_review": "ಕಳಪೆ ವಿಮರ್ಶೆಗಳ ಶೇಕಡಾವಾರು:",
        },
        "en": {
            "medicine_name": "Medicine name:",
            "uses": "Recorded uses:",
            "side_effects": "Recorded possible side effects:",
            "composition": "Composition:",
            "manufacturer": "Manufacturer:",
            "excellent_review": "Excellent review percentage:",
            "average_review": "Average review percentage:",
            "poor_review": "Poor review percentage:",
        },
    }
    if intent == "dosage":
        answer = (
            "ಈ ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಡೋಸ್ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ವೈದ್ಯರು ಅಥವಾ ಔಷಧ ತಜ್ಞರು ನೀಡಿದ ಸೂಚನೆಯನ್ನು ಅನುಸರಿಸಿ."
            if language == "kn"
            else "Dosage information is unavailable in this database. Follow the instructions from your doctor or pharmacist."
        )
        fields = []
    elif intent == "reviews":
        if language == "kn":
            answer = (
                f"ಉತ್ತಮ ವಿಮರ್ಶೆಗಳು: {row.get('Excellent Review %','')}%. "
                f"ಸರಾಸರಿ ವಿಮರ್ಶೆಗಳು: {row.get('Average Review %','')}%. "
                f"ಕಳಪೆ ವಿಮರ್ಶೆಗಳು: {row.get('Poor Review %','')}%."
            )
        else:
            answer = (
                f"Excellent reviews: {row.get('Excellent Review %','')}%. "
                f"Average reviews: {row.get('Average Review %','')}%. "
                f"Poor reviews: {row.get('Poor Review %','')}%."
            )
        fields = ["excellent_review", "average_review", "poor_review"]
    elif intent in values:
        value = values[intent].strip()
        missing = (
            "ಈ ಮಾಹಿತಿಯು ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ."
            if language == "kn"
            else "This information is unavailable in the database."
        )
        answer = f"{prefixes[language][intent]} {value}" if value else missing
        fields = [intent]
    elif language == "kn":
        answer = (
            f"ಔಷಧಿ: {row['Medicine Name']}. ಸಂಯೋಜನೆ: {row.get('Composition','')}. "
            f"ಬಳಕೆಗಳು: {row.get('Uses','')}. ಸಂಭವನೀಯ ಅಡ್ಡ ಪರಿಣಾಮಗಳು: {row.get('Side_effects','')}. "
            f"ತಯಾರಕರು: {row.get('Manufacturer','')}."
        )
        fields = ["composition", "uses", "side_effects", "manufacturer"]
    else:
        answer = (
            f"Medicine: {row['Medicine Name']}. Composition: {row.get('Composition','')}. "
            f"Uses: {row.get('Uses','')}. Possible side effects: {row.get('Side_effects','')}. "
            f"Manufacturer: {row.get('Manufacturer','')}."
        )
        fields = ["composition", "uses", "side_effects", "manufacturer"]
    generated = False
    generation_error = None
    llm_output = None
    # The image classifier has already selected one exact CSV record. Send that
    # complete record as grounded context; never send the image or other rows.
    grounded_evidence = {
        field: value for field, value in row.items() if value not in (None, "")
    }
    if generator is not None:
        try:
            answer = generator.answer(row["Medicine Name"], question, grounded_evidence)
            llm_output = answer
            generated = True
        except Exception as exc:
            generation_error = str(exc)
    return {
        "status": "success",
        "answer": answer,
        "language": language,
        "intent": intent,
        "medicine_name": row["Medicine Name"],
        "evidence": grounded_evidence,
        "llm_output": llm_output,
        "source": {"type": "local_csv", "database": str(DEFAULT_DATABASE.name)},
        "match": {"type": resolution["match_type"], "score": resolution["score"]},
        "generation": {
            "backend": (
                getattr(generator, "backend", "external")
                if generator is not None
                else "template"
            ),
            "generated": generated,
            "fallback_reason": generation_error,
            "model": getattr(generator, "last_model", None) if generated else None,
        },
    }
