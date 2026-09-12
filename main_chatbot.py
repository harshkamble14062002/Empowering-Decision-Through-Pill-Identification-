#!/usr/bin/env python3
"""Full-image OCR/CSV identification followed by a grounded bilingual answer."""
import argparse
import json
from pathlib import Path
import sys

from kannada_rag import DEFAULT_INDEX, answer_kannada, build_index, detect_language
from openrouter_llm import OpenRouterGenerator
from ocr_csv_classifier import DEFAULT_DATABASE, run_ocr_csv_pipeline
DEFAULT_MULTICLASS_MODEL = str(Path(__file__).resolve().parent / "multiclass_models" / "best_multiclass.pt")


def run_multiclass_pipeline(*args, **kwargs):
    """Load the optional YOLO comparison pipeline only when explicitly requested."""
    from pipeline_multiclass import run_multiclass_pipeline as implementation
    return implementation(*args, **kwargs)


def run(image, question, database=DEFAULT_DATABASE, index=DEFAULT_INDEX,
        output="output", quiet=False, generator=None,
        identification_backend="ocr_csv", model=DEFAULT_MULTICLASS_MODEL):
    index = Path(index)
    if not index.exists():
        build_index(database, index)
    if identification_backend == "ocr_csv":
        identification = run_ocr_csv_pipeline(
            image, database_path=database, output_dir=output, quiet=quiet
        )
    elif identification_backend == "multiclass":
        identification = run_multiclass_pipeline(
            image, model_path=model, database_path=database,
            output_dir=output, quiet=quiet,
        )
    else:
        raise ValueError(f"Unknown identification backend: {identification_backend}")
    if identification["status"] != "success":
        return {
            "status": "identification_uncertain",
            "identification": identification,
            "chatbot": {
                "status": "withheld",
                "answer": (
                    "ಔಷಧಿಯನ್ನು ಖಚಿತವಾಗಿ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ಆದ್ದರಿಂದ ಔಷಧಿ ಮಾಹಿತಿಯನ್ನು ನೀಡಲಾಗುವುದಿಲ್ಲ."
                    if detect_language(question) == "kn" else
                    "The medicine could not be identified confidently, so its information cannot be provided."
                ),
                "evidence": None,
            },
        }
    medicine_name = identification["final_result"]["medicine_name"]
    chatbot = answer_kannada(medicine_name, question, index, generator)
    return {
        "status": "success" if chatbot["status"] == "success" else chatbot["status"],
        "identification": identification,
        "chatbot": chatbot,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--identification-backend", choices=("ocr_csv", "multiclass"), default="ocr_csv")
    parser.add_argument("--model", default=DEFAULT_MULTICLASS_MODEL)
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--output", default="output")
    parser.add_argument("--save-json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use deterministic templates instead of OpenRouter")
    parser.add_argument("--openrouter-model", default="openrouter/free")
    args = parser.parse_args()
    generator = None if args.no_llm else OpenRouterGenerator(model=args.openrouter_model)
    result = run(
        args.image, args.question, database=args.database, index=args.index,
        output=args.output, quiet=args.quiet, generator=generator,
        identification_backend=args.identification_backend, model=args.model
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.save_json:
        Path(args.save_json).write_text(rendered)
    sys.exit(0 if result["status"] == "success" else 2)


if __name__ == "__main__":
    main()
