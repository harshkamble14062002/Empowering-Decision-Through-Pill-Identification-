"""Final conservative gate, including joined brand-variant suffix checks."""

from single_class_safety_v2 import (
    VARIANT_MARKERS,
    tokens,
    verify_single_candidate as verify_v2,
)


def verify_single_candidate(ocr_text, ocr_confidence, candidate):
    decision = verify_v2(ocr_text, ocr_confidence, candidate)
    if decision["status"] != "success" or not candidate:
        return decision

    name = candidate.get("medicine_name") or candidate.get("Medicine Name") or ""
    name_tokens = tokens(name)
    name_markers = set(name_tokens) & VARIANT_MARKERS
    candidate_words = [word for word in name_tokens if word.isalpha()]

    for observed in [word for word in tokens(ocr_text) if word.isalpha()]:
        for candidate_word in candidate_words:
            if observed.startswith(candidate_word) and observed != candidate_word:
                suffix = observed[len(candidate_word):]
                if suffix in VARIANT_MARKERS and suffix not in name_markers:
                    decision["status"] = "uncertain"
                    decision["reasons"].append("joined_candidate_variant_conflict")
                    return decision
    return decision

