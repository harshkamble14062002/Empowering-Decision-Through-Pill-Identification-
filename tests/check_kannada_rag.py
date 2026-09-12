"""Dependency-free Sprint 4 smoke and behavior checks."""
from pathlib import Path
import tempfile

from test_kannada_rag import (
    test_dosage_is_not_invented,
    test_grounded_kannada_use_answer,
    test_kannada_intents,
    test_retrieval_uses_index,
    test_unknown_medicine_is_not_answered,
)


def main():
    test_kannada_intents()
    with tempfile.TemporaryDirectory() as directory:
        test_grounded_kannada_use_answer(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_dosage_is_not_invented(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_unknown_medicine_is_not_answered(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_retrieval_uses_index(Path(directory))
    print("5 Kannada RAG checks passed")


if __name__ == "__main__":
    main()
