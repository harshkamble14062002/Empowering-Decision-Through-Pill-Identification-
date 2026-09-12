"""Retrieve detailed U.S. drug-label evidence from the official openFDA API."""
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

from rapidfuzz import fuzz
import requests


ENDPOINT = "https://api.fda.gov/drug/label.json"
ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "api_cache" / "openfda"

FIELDS = {
    "uses": ["indications_and_usage", "purpose"],
    "dosage": ["dosage_and_administration"],
    "warnings": ["boxed_warning", "warnings", "warnings_and_cautions"],
    "contraindications": ["contraindications", "do_not_use"],
    "interactions": ["drug_interactions", "ask_doctor_or_pharmacist"],
    "side_effects": ["adverse_reactions", "when_using"],
    "pregnancy": ["pregnancy", "pregnancy_or_breast_feeding"],
    "ingredients": ["active_ingredient", "inactive_ingredient", "description"],
    "storage": ["storage_and_handling", "spl_unclassified_section"],
}

INTENT_TERMS = {
    "dosage": ["dose", "dosage", "how much", "ಡೋಸ್", "ಪ್ರಮಾಣ", "ಎಷ್ಟು ತೆಗೆದುಕೊಳ್ಳ"],
    "warnings": ["warning", "precaution", "ಎಚ್ಚರಿಕೆ", "ಮುನ್ನೆಚ್ಚರಿಕೆ"],
    "contraindications": ["contraindication", "do not use", "ಯಾರು ಬಳಸಬಾರದು", "ಬಳಸಬಾರದು"],
    "interactions": ["interaction", "other medicine", "ಪರಸ್ಪರ ಕ್ರಿಯೆ", "ಬೇರೆ ಔಷಧ"],
    "pregnancy": ["pregnant", "pregnancy", "breastfeeding", "ಗರ್ಭಧಾರಣೆ", "ಸ್ತನ್ಯಪಾನ"],
    "side_effects": ["side effect", "adverse", "ಅಡ್ಡ ಪರಿಣಾಮ", "ದುಷ್ಪರಿಣಾಮ"],
    "ingredients": ["ingredient", "composition", "contains", "ಸಂಯೋಜನೆ", "ಘಟಕ"],
    "storage": ["storage", "store", "ಸಂಗ್ರಹ", "ಇಡಬೇಕು"],
    "uses": ["use", "used for", "purpose", "ಉಪಯೋಗ", "ಬಳಕೆ", "ಯಾವುದಕ್ಕೆ", "ಏಕೆ"],
}


def label_intent(question):
    lowered = (question or "").lower()
    scores = {intent: sum(term in lowered for term in terms) for intent, terms in INTENT_TERMS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else None


def ingredients_from_composition(composition):
    text = re.sub(r"\([^)]*\)", " ", composition or "")
    return [part.strip() for part in re.split(r"\s*\+\s*|\s+and\s+", text, flags=re.I) if part.strip()]


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _first(record, keys):
    values = []
    for key in keys:
        value = record.get(key, [])
        values.extend(value if isinstance(value, list) else [value])
    return "\n".join(str(value).strip() for value in values if str(value).strip())


def _candidate_score(record, ingredients, composition):
    openfda = record.get("openfda", {})
    substances = " ".join(openfda.get("substance_name", []))
    active = _first(record, ["active_ingredient", "spl_product_data_elements"])
    searchable = substances + " " + active
    word_scores = [fuzz.partial_ratio(ingredient.lower(), searchable.lower()) for ingredient in ingredients]
    if not word_scores or min(word_scores) < 72:
        return 0
    expected_numbers = _numbers(composition)
    observed_numbers = _numbers(active)
    if expected_numbers and observed_numbers and not expected_numbers.issubset(observed_numbers):
        return 0
    return sum(word_scores) / len(word_scores)


def fetch_openfda_label(composition, api_key=None, cache_dir=DEFAULT_CACHE, timeout=20):
    ingredients = ingredients_from_composition(composition)
    if not ingredients:
        return {"status": "unavailable", "reason": "missing_composition", "label": None}
    query_ingredient = ingredients[0]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(composition.lower().encode()).hexdigest()
    cache_file = cache_dir / (cache_key + ".json")
    params = {"search": f'openfda.substance_name:"{query_ingredient}"', "limit": 10}
    key = api_key or os.environ.get("OPENFDA_API_KEY")
    if key:
        params["api_key"] = key
    try:
        if cache_file.exists():
            payload = json.loads(cache_file.read_text())
            source = "cache"
        else:
            response = requests.get(ENDPOINT, params=params, timeout=timeout)
            if response.status_code == 404:
                return {"status": "no_match", "reason": "no_openfda_label", "label": None}
            response.raise_for_status()
            payload = response.json()
            cache_file.write_text(json.dumps(payload))
            source = "openfda"
    except (requests.RequestException, ValueError, OSError) as error:
        return {"status": "unavailable", "reason": "openfda_request_failed",
                "error": str(error), "label": None}
    ranked = sorted(
        ((_candidate_score(record, ingredients, composition), record) for record in payload.get("results", [])),
        key=lambda item: item[0], reverse=True
    )
    if not ranked or ranked[0][0] < 72:
        return {"status": "uncertain", "reason": "label_does_not_match_composition", "label": None}
    score, record = ranked[0]
    openfda = record.get("openfda", {})
    set_id = record.get("set_id") or (openfda.get("spl_set_id") or [None])[0]
    return {
        "status": "success",
        "match_score": round(float(score), 2),
        "jurisdiction": "United States FDA product label",
        "source": source,
        "source_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else ENDPOINT,
        "set_id": set_id,
        "effective_time": record.get("effective_time"),
        "generic_name": openfda.get("generic_name", []),
        "substance_name": openfda.get("substance_name", []),
        "label": {intent: _first(record, keys) for intent, keys in FIELDS.items()},
    }


def answer_from_openfda(details, question):
    intent = label_intent(question)
    if details.get("status") != "success" or not intent:
        return None
    evidence = details["label"].get(intent, "")
    if not evidence:
        return {
            "status": "information_unavailable", "intent": intent,
            "answer": "ಈ ಮಾಹಿತಿಯು ಹೊಂದಾಣಿಕೆಯ ಅಧಿಕೃತ ಲೇಬಲ್‌ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ.",
            "evidence": None, "source_url": details["source_url"],
        }
    prefixes = {
        "uses": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಬಳಕೆ ಮಾಹಿತಿ:",
        "dosage": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಡೋಸ್ ಮಾಹಿತಿ:",
        "warnings": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಎಚ್ಚರಿಕೆಗಳು:",
        "contraindications": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್ ಪ್ರಕಾರ ಬಳಸಬಾರದ ಸಂದರ್ಭಗಳು:",
        "interactions": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಔಷಧ ಪರಸ್ಪರ ಕ್ರಿಯೆಗಳು:",
        "side_effects": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಅಡ್ಡ ಪರಿಣಾಮಗಳು:",
        "pregnancy": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಗರ್ಭಧಾರಣೆ/ಸ್ತನ್ಯಪಾನ ಮಾಹಿತಿ:",
        "ingredients": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಪದಾರ್ಥಗಳು:",
        "storage": "ಅಮೆರಿಕದ FDA ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಸಂಗ್ರಹ ಮಾಹಿತಿ:",
    }
    return {
        "status": "success", "intent": intent,
        "answer": prefixes[intent] + " " + evidence,
        "evidence": evidence,
        "jurisdiction": details["jurisdiction"],
        "effective_time": details["effective_time"],
        "source_url": details["source_url"],
    }
