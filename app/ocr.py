"""Read medicine packaging and match its text against the catalogue."""

import csv
import re
import time
from pathlib import Path

import cv2
from rapidfuzz import fuzz

import threading

import easyocr
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data" / "medicines.csv"
IGNORE_PHRASES = {
    "mrp",
    "batch",
    "batch no",
    "expiry",
    "expiry date",
    "mfg date",
    "manufacturing date",
    "use before",
    "schedule h",
    "keep out of reach",
    "store below",
    "for external use",
    "not for sale",
    "sample",
}
GENERIC_TOKENS = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "release",
    "sustained",
    "extended",
    "film",
    "coated",
    "strip",
    "strips",
    "ip",
    "usp",
    "bp",
    "contains",
    "each",
    "dosage",
    "colour",
    "warning",
    "rx",
    "only",
}
ALIASES = {"limited": "ltd", "private": "pvt", "laboratories": "labs"}
FORMULATION = {"sr", "cr", "xr", "er", "mr", "od", "plus", "forte", "n"}
DOSAGE_UNITS = {"mg", "mcg", "g", "ml", "iu"}


_READERS = {}
_READER_LOCK = threading.RLock()


def _read_ocr(image, lang):
    # Share model initialization and serialize access to the GPU reader.
    with _READER_LOCK:
        key = tuple(lang)
        if key not in _READERS:
            import torch

            try:
                _READERS[key] = easyocr.Reader(
                    list(lang), gpu=torch.cuda.is_available()
                )
            except Exception:
                _READERS[key] = easyocr.Reader(list(lang), gpu=False)
        return _READERS[key].readtext(image)


def normalize(text):
    values = re.findall(r"[a-z]+|\d+(?:\.\d+)?", (text or "").lower())
    return " ".join(ALIASES.get(value, value) for value in values)


def tokens(text):
    return normalize(text).split()


def is_irrelevant(text):
    value = normalize(text)
    if not value:
        return True
    if any(phrase in value for phrase in IGNORE_PHRASES):
        return True
    if re.fullmatch(r"\d+\s*x\s*\d+(?:\s+(?:tablet|tablets|capsule|capsules))?", value):
        return True
    meaningful = [item for item in tokens(value) if item not in GENERIC_TOKENS]
    return not meaningful


def similarity(observed, expected):
    left, right = normalize(observed), normalize(expected)
    if not left or not right:
        return 0.0
    compact_left, compact_right = left.replace(" ", ""), right.replace(" ", "")
    return max(
        fuzz.ratio(compact_left, compact_right),
        fuzz.token_sort_ratio(left, right),
        fuzz.token_set_ratio(left, right),
    )


def brand_root(medicine_name):
    """Return the leading brand word, excluding dosage and packaging tokens."""
    for token in tokens(medicine_name):
        if (
            token not in GENERIC_TOKENS | FORMULATION | DOSAGE_UNITS
            and not re.fullmatch(r"\d+(?:\.\d+)?", token)
        ):
            return token
    return ""


def brand_evidence(line, medicine_name):
    """Score a line only when it contains the candidate's actual brand root.

    This prevents isolated strengths such as ``75`` or ``5 mg`` and
    composition text from becoming perfect brand matches for many CSV rows.
    """
    root = brand_root(medicine_name)
    words = [
        token
        for token in tokens(line["text"])
        if token not in GENERIC_TOKENS | DOSAGE_UNITS and re.search(r"[a-z]", token)
    ]
    identity = max((fuzz.ratio(root, word) for word in words), default=0.0)
    if not root or identity < 65:
        return 0.0
    return 0.70 * identity + 0.30 * similarity(line["text"], medicine_name)


def same_line_groups(raw):
    """Join OCR boxes on the same visual line, such as `Acemiz` + `200 SR`."""
    items = []
    for bbox, text, confidence in raw:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        items.append(
            {
                "text": text.strip(),
                "confidence": float(confidence),
                "x1": min(xs),
                "x2": max(xs),
                "y1": min(ys),
                "y2": max(ys),
                "cy": sum(ys) / len(ys),
            }
        )
    items.sort(key=lambda item: (item["cy"], item["x1"]))
    lines = []
    for item in items:
        height = max(1, item["y2"] - item["y1"])
        line = next(
            (
                line
                for line in lines
                if abs(line["cy"] - item["cy"]) <= 0.55 * max(height, line["height"])
            ),
            None,
        )
        if line is None:
            lines.append({"items": [item], "cy": item["cy"], "height": height})
        else:
            line["items"].append(item)
            line["cy"] = sum(part["cy"] for part in line["items"]) / len(line["items"])
            line["height"] = max(line["height"], height)
    output = []
    for line in lines:
        parts = sorted(line["items"], key=lambda item: item["x1"])
        output.append(
            {
                "text": " ".join(part["text"] for part in parts),
                "confidence": sum(part["confidence"] for part in parts) / len(parts),
                "bbox": [
                    float(min(p["x1"] for p in parts)),
                    float(min(p["y1"] for p in parts)),
                    float(max(p["x2"] for p in parts)),
                    float(max(p["y2"] for p in parts)),
                ],
            }
        )
    return output


def preclassify_lines(lines, rows):
    """Assign each OCR line to one field globally before ranking medicine rows."""
    values = {
        "brand_name": list({row["Medicine Name"] for row in rows}),
        "composition": list(
            {row.get("Composition", "") for row in rows if row.get("Composition")}
        ),
        "manufacturer": list(
            {row.get("Manufacturer", "") for row in rows if row.get("Manufacturer")}
        ),
    }
    manufacturer_tokens = [set(tokens(value)) for value in values["manufacturer"]]
    output = []
    for line in lines:
        text_tokens = set(tokens(line["text"]))
        scores = {
            name: max(
                (similarity(line["text"], value) for value in candidates), default=0
            )
            for name, candidates in values.items()
        }
        manufacturer_identity = (
            bool(text_tokens)
            and any(
                text_tokens.issubset(candidate) for candidate in manufacturer_tokens
            )
            and scores["manufacturer"] >= 85
        )
        if is_irrelevant(line["text"]):
            class_name = "ignore"
        elif manufacturer_identity:
            class_name = "manufacturer"
        elif (
            scores["composition"] >= 65
            and scores["composition"] >= scores["brand_name"] + 5
        ):
            class_name = "composition"
        elif scores["brand_name"] >= 68:
            class_name = "brand_name"
        elif scores["composition"] >= 65:
            class_name = "composition"
        else:
            class_name = "ignore"
        output.append(
            {
                **line,
                "preliminary_class": class_name,
                "global_scores": {
                    key: round(value, 2) for key, value in scores.items()
                },
            }
        )
    return output


def rank_rows(lines, rows):
    by_class = {
        name: [line for line in lines if line["preliminary_class"] == name]
        for name in ("brand_name", "composition", "manufacturer")
    }
    brand_lines = [
        line
        for line in lines
        if line["preliminary_class"] not in {"composition", "manufacturer"}
        and not is_irrelevant(line["text"])
    ]
    ranked = []
    for row in rows:
        brand_options = [
            (brand_evidence(line, row["Medicine Name"]), line) for line in brand_lines
        ]
        comp_options = [
            (similarity(line["text"], row.get("Composition", "")), line)
            for line in by_class["composition"]
        ]
        mfg_options = [
            (similarity(line["text"], row.get("Manufacturer", "")), line)
            for line in by_class["manufacturer"]
        ]
        brand_score, brand_line = max(
            brand_options, default=(0, {"text": ""}), key=lambda item: item[0]
        )
        comp_score, _ = max(comp_options, default=(0, None), key=lambda item: item[0])
        mfg_score, _ = max(mfg_options, default=(0, None), key=lambda item: item[0])
        observed = set(re.findall(r"\d+(?:\.\d+)?", normalize(brand_line["text"])))
        expected = set(re.findall(r"\d+(?:\.\d+)?", normalize(row["Medicine Name"])))
        penalty = 0
        if observed and not observed.issubset(expected):
            penalty += 25
        observed_forms = set(tokens(brand_line["text"])) & FORMULATION
        expected_forms = set(tokens(row["Medicine Name"])) & FORMULATION
        if observed_forms and not observed_forms.issubset(expected_forms):
            penalty += 10
        total = 0.60 * brand_score + 0.30 * comp_score + 0.10 * mfg_score - penalty
        ranked.append(
            {
                "row": row,
                "score": total,
                "brand": brand_score,
                "composition": comp_score,
                "manufacturer": mfg_score,
                "brand_text": brand_line["text"],
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def classify_lines(lines, row):
    output = []
    fields = {
        "brand_name": row["Medicine Name"],
        "composition": row.get("Composition", ""),
        "manufacturer": row.get("Manufacturer", ""),
    }
    thresholds = {"brand_name": 68, "composition": 60, "manufacturer": 70}
    for line in lines:
        scores = {
            name: similarity(line["text"], expected)
            for name, expected in fields.items()
        }
        name = line["preliminary_class"]
        if name == "ignore" or scores[name] < thresholds[name]:
            name = "ignore"
        output.append(
            {key: value for key, value in line.items() if key != "preliminary_class"}
            | {
                "normalized_text": normalize(line["text"]),
                "class_name": name,
                "scores": {key: round(value, 2) for key, value in scores.items()},
            }
        )
    return output


def analyze(image_path, database_path=DEFAULT_DATABASE, output_path=None):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    raw = _read_ocr(image, ["en"])
    raw = [item for item in raw if float(item[2]) >= 0.25]
    lines = same_line_groups(raw)
    with open(database_path, newline="", errors="replace") as stream:
        rows = list(csv.DictReader(stream))
    lines = preclassify_lines(lines, rows)
    ranked = rank_rows(lines, rows)
    best = ranked[0]
    margin = best["score"] - ranked[1]["score"] if len(ranked) > 1 else best["score"]
    combined_evidence = best["score"] >= 70 and best["brand"] >= 68 and margin >= 4
    strong_unique_brand = best["brand"] >= 88 and margin >= 6
    accepted = combined_evidence or strong_unique_brand
    acceptance_policy = (
        "combined_evidence"
        if combined_evidence
        else "strong_unique_brand" if strong_unique_brand else None
    )
    classifications = (
        classify_lines(lines, best["row"])
        if accepted
        else [
            {key: value for key, value in line.items() if key != "preliminary_class"}
            | {
                "normalized_text": normalize(line["text"]),
                "class_name": "ignore",
                "scores": {},
            }
            for line in lines
        ]
    )
    result = {
        "status": "success" if accepted else "uncertain",
        "medicine": best["row"] if accepted else None,
        "score": round(best["score"], 2),
        "margin": round(margin, 2),
        "field_scores": {
            key: round(best[key], 2) for key in ("brand", "composition", "manufacturer")
        },
        "acceptance_policy": acceptance_policy,
        "ocr_regions": classifications,
        "candidates": [
            {
                "medicine_name": item["row"]["Medicine Name"],
                "score": round(item["score"], 2),
            }
            for item in ranked[:3]
        ],
    }
    if output_path:
        colors = {
            "brand_name": (0, 180, 0),
            "composition": (255, 150, 0),
            "manufacturer": (0, 0, 220),
            "ignore": (150, 150, 150),
        }
        preview = image.copy()
        for item in classifications:
            x1, y1, x2, y2 = map(int, item["bbox"])
            color = colors[item["class_name"]]
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                preview,
                item["class_name"],
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), preview)
        result["annotated_path"] = str(output_path)
    return result


def run_ocr_csv_pipeline(
    image_path, database_path=DEFAULT_DATABASE, output_dir="output", quiet=False
):
    """Return the matched medicine, OCR boxes and any uncertainty reasons."""
    started = time.time()
    output_dir = Path(output_dir)
    annotated_path = output_dir / "annotated" / f"{Path(image_path).stem}_ocr_csv.jpg"
    classified = analyze(image_path, database_path, annotated_path)
    row = classified.get("medicine")
    reasons = []
    if classified["field_scores"]["brand"] < 68:
        reasons.append("weak_brand_identity")
    if classified["score"] < 70:
        reasons.append("weak_combined_match")
    if classified["margin"] < 4:
        reasons.append("ambiguous_candidates")
    success = classified["status"] == "success" and row is not None
    final = None
    if success:
        final = {
            "medicine_name": row.get("Medicine Name"),
            "composition": row.get("Composition"),
            "manufacturer": row.get("Manufacturer"),
            "uses": row.get("Uses"),
            "side_effects": row.get("Side_effects"),
            "match_score": classified["score"],
            "match_strategy": "full_image_ocr_csv",
        }
    return {
        "image_path": str(image_path),
        "status": "success" if success else "uncertain",
        "final_result": final,
        "match_strategy": "full_image_ocr_csv",
        "uncertainty_reasons": [] if success else reasons,
        "candidates": classified["candidates"],
        "timings": {"total_identification": time.time() - started},
        "stages": {
            "detection": {
                "method": "full_image_easyocr",
                "annotated_path": str(annotated_path),
            },
            "ocr_csv": classified,
        },
    }
