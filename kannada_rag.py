"""Grounded local medicine retrieval and Kannada answer generation."""
from functools import lru_cache
import argparse
import csv
import json
from pathlib import Path
import re

import joblib
from rapidfuzz import fuzz, process


ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / "Medicine_Details_Verified.csv"
DEFAULT_INDEX = ROOT / "medicine_rag_index.joblib"
DEFAULT_SEMANTIC_INDEX = ROOT / "medicine_rag_sbert_index.joblib"


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def document(row):
    return " | ".join(
        str(row.get(field, ""))
        for field in ("Medicine Name", "Composition", "Uses", "Side_effects", "Manufacturer")
    )


def build_index(database=DEFAULT_DATABASE, destination=DEFAULT_INDEX):
    database, destination = Path(database).resolve(), Path(destination).resolve()
    with database.open(newline="", errors="replace") as stream:
        rows = list(csv.DictReader(stream))
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(document(row) for row in rows)
    payload = {
        "database": str(database),
        "database_sha256": __import__("hashlib").sha256(database.read_bytes()).hexdigest(),
        "rows": rows,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }
    joblib.dump(payload, destination, compress=3)
    return {"records": len(rows), "dimensions": matrix.shape[1], "index": str(destination)}


def build_semantic_index(database=DEFAULT_DATABASE, destination=DEFAULT_SEMANTIC_INDEX,
                         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Build a multilingual Sentence-BERT index without changing the source CSV."""
    from sentence_transformers import SentenceTransformer

    database, destination = Path(database).resolve(), Path(destination).resolve()
    with database.open(newline="", errors="replace") as stream:
        rows = list(csv.DictReader(stream))
    model = SentenceTransformer(model_name)
    matrix = model.encode(
        [document(row) for row in rows], batch_size=64,
        show_progress_bar=True, normalize_embeddings=True,
    )
    payload = {
        "backend": "sentence_transformers",
        "model_name": model_name,
        "database": str(database),
        "database_sha256": __import__("hashlib").sha256(database.read_bytes()).hexdigest(),
        "rows": rows,
        "matrix": matrix,
    }
    joblib.dump(payload, destination, compress=3)
    return {"records": len(rows), "dimensions": matrix.shape[1], "index": str(destination),
            "model": model_name}


@lru_cache(maxsize=2)
def load_index(path, modified):
    return joblib.load(path)


def retrieve(query, index=DEFAULT_INDEX, limit=3):
    from sklearn.metrics.pairwise import cosine_similarity

    path = Path(index).resolve()
    payload = load_index(str(path), path.stat().st_mtime_ns)
    if payload.get("backend") == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(payload["model_name"])
        query_vector = model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(query_vector, payload["matrix"]).ravel()
    else:
        query_vector = payload["vectorizer"].transform([query])
        scores = cosine_similarity(query_vector, payload["matrix"]).ravel()
    order = scores.argsort()[::-1][:limit]
    return [
        {"score": round(float(scores[i]), 4), "medicine": payload["rows"][int(i)]}
        for i in order
    ]


def resolve_confirmed_medicine(name, index=DEFAULT_INDEX):
    path = Path(index).resolve()
    payload = load_index(str(path), path.stat().st_mtime_ns)
    names = [row["Medicine Name"] for row in payload["rows"]]
    normalized = [normalize(value) for value in names]
    query = normalize(name)
    exact = [i for i, value in enumerate(normalized) if value == query]
    if len(exact) == 1:
        return {"status": "success", "match_type": "exact", "score": 100.0,
                "medicine": payload["rows"][exact[0]], "candidates": []}
    ranked = process.extract(query, normalized, scorer=fuzz.ratio, limit=3)
    candidates = [
        {"medicine_name": names[index_value], "score": round(float(score), 2)}
        for _, score, index_value in ranked
    ]
    if not ranked or ranked[0][1] < 88:
        return {"status": "no_match", "medicine": None, "candidates": candidates,
                "reasons": ["confirmed_medicine_not_found"]}
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 5:
        return {"status": "uncertain", "medicine": None, "candidates": candidates,
                "reasons": ["ambiguous_medicine_name"]}
    return {"status": "success", "match_type": "fuzzy", "score": round(float(ranked[0][1]), 2),
            "medicine": payload["rows"][ranked[0][2]], "candidates": candidates}


INTENT_TERMS = {
    "medicine_name": ["medicine name", "name of this medicine", "ಔಷಧಿಯ ಹೆಸರು"],
    "uses": ["use", "uses", "used for", "benefit", "ಉಪಯೋಗ", "ಬಳಕೆ", "ಯಾವುದಕ್ಕೆ", "ಏಕೆ", "ಯಾವ ಕಾಯಿಲೆ"],
    "side_effects": ["side effect", "adverse", "ಅಡ್ಡ ಪರಿಣಾಮ", "ದುಷ್ಪರಿಣಾಮ", "ವಾಂತಿ", "ತಲೆನೋವು"],
    "composition": ["composition", "ingredient", "contains", "strength", "ಸಂಯೋಜನೆ", "ಘಟಕ", "ಪದಾರ್ಥ", "ಶಕ್ತಿ"],
    "manufacturer": ["manufacturer", "company", "made by", "ತಯಾರಕ", "ಕಂಪನಿ"],
    "excellent_review": ["excellent review", "good review", "ಉತ್ತಮ ವಿಮರ್ಶೆ"],
    "average_review": ["average review", "ಸರಾಸರಿ ವಿಮರ್ಶೆ"],
    "poor_review": ["poor review", "bad review", "ಕಳಪೆ ವಿಮರ್ಶೆ"],
    "reviews": ["all review", "review percentage", "ಎಲ್ಲಾ ವಿಮರ್ಶಾ", "ವಿಮರ್ಶಾ ಶೇಕಡಾವಾರು"],
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
            if language == "kn" else
            "The medicine could not be identified confidently. Please check its complete name."
        )
        return {
            "status": resolution["status"], "answer": answer, "language": language,
            "evidence": None, "candidates": resolution.get("candidates", []),
            "reasons": resolution.get("reasons", []),
        }
    row = resolution["medicine"]
    intent = detect_intent(question)
    values = {
        "medicine_name": row.get("Medicine Name", ""),
        "uses": row.get("Uses", ""), "side_effects": row.get("Side_effects", ""),
        "composition": row.get("Composition", ""), "manufacturer": row.get("Manufacturer", ""),
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
            "uses": "Recorded uses:", "side_effects": "Recorded possible side effects:",
            "composition": "Composition:", "manufacturer": "Manufacturer:",
            "excellent_review": "Excellent review percentage:",
            "average_review": "Average review percentage:",
            "poor_review": "Poor review percentage:",
        },
    }
    if intent == "dosage":
        answer = (
            "ಈ ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಡೋಸ್ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ವೈದ್ಯರು ಅಥವಾ ಔಷಧ ತಜ್ಞರು ನೀಡಿದ ಸೂಚನೆಯನ್ನು ಅನುಸರಿಸಿ."
            if language == "kn" else
            "Dosage information is unavailable in this database. Follow the instructions from your doctor or pharmacist."
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
        missing = "ಈ ಮಾಹಿತಿಯು ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ." if language == "kn" else "This information is unavailable in the database."
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
    grounded_evidence = {field: value for field, value in row.items() if value not in (None, "")}
    if generator is not None:
        try:
            answer = generator.answer(row["Medicine Name"], question, grounded_evidence)
            llm_output = answer
            generated = True
        except Exception as exc:
            generation_error = str(exc)
    return {
        "status": "success", "answer": answer, "language": language,
        "intent": intent, "medicine_name": row["Medicine Name"],
        "evidence": grounded_evidence,
        "llm_output": llm_output,
        "source": {"type": "local_csv", "database": str(DEFAULT_DATABASE.name)},
        "match": {"type": resolution["match_type"], "score": resolution["score"]},
        "generation": {
            "backend": getattr(generator, "backend", "external") if generator is not None else "template",
            "generated": generated, "fallback_reason": generation_error,
            "model": getattr(generator, "last_model", None) if generated else None,
        },
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--build-semantic-index", action="store_true")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--openrouter", action="store_true",
                        help="Generate Kannada with OPENROUTER_API_KEY and openrouter/free")
    parser.add_argument("--openrouter-model", default="openrouter/free")
    parser.add_argument("--medicine")
    parser.add_argument("--question")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.build_semantic_index:
        print(json.dumps(build_semantic_index(args.database, args.index, args.embedding_model), indent=2))
        if not args.medicine:
            return
    elif args.build_index or not args.index.exists():
        print(json.dumps(build_index(args.database, args.index), indent=2))
        if not args.medicine:
            return
    if not args.medicine or not args.question:
        parser.error("--medicine and --question are required for a question")
    generator = None
    if args.openrouter:
        from openrouter_llm import OpenRouterGenerator
        generator = OpenRouterGenerator(model=args.openrouter_model)
    result = answer_kannada(args.medicine, args.question, args.index, generator)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["answer"])


if __name__ == "__main__":
    main()
