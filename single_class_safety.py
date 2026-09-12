"""Safety gate for candidates returned by the legacy single-class pipeline."""

import re
from rapidfuzz import fuzz


GENERIC = {
    "tablet", "tablets", "capsule", "capsules", "soft", "gelatin",
    "mg", "mcg", "ml", "gm", "ip", "bp", "usp",
}


def tokens(text):
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", (text or "").lower())


def verify_single_candidate(ocr_text, ocr_confidence, candidate):
    """Accept only when OCR directly supports the returned name and strength."""
    if not candidate:
        return {
            "status": "no_match",
            "reasons": ["no_candidate"],
            "candidate": None,
        }

    name = candidate.get("medicine_name") or candidate.get("Medicine Name") or ""
    score = float(candidate.get("match_score", 0) or 0)
    observed = tokens(ocr_text)
    observed_words = [token for token in observed if token.isalpha()]
    name_tokens = tokens(name)
    name_words = [
        token for token in name_tokens
        if token.isalpha() and token not in GENERIC
    ]
    name_numbers = {token for token in name_tokens if token[0].isdigit()}
    observed_numbers = {token for token in observed if token[0].isdigit()}

    reasons = []
    if ocr_confidence < 0.60:
        reasons.append("low_ocr_confidence")
    if score < 75:
        reasons.append("low_lookup_score")
    if not name_words:
        reasons.append("missing_candidate_brand")
    else:
        for word in name_words:
            if len(word) <= 2:
                supported = word in observed_words
            else:
                supported = any(fuzz.ratio(word, item) >= 80 for item in observed_words)
            if not supported:
                reasons.append("candidate_brand_not_read")
                break
    if name_numbers and not name_numbers.issubset(observed_numbers):
        reasons.append("candidate_strength_not_read")

    return {
        "status": "uncertain" if reasons else "success",
        "reasons": reasons,
        "candidate": candidate,
    }

