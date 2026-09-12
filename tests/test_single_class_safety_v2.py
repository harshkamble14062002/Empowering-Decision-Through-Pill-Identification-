import unittest

from single_class_safety_v2 import verify_single_candidate


class SingleClassSafetyV2Tests(unittest.TestCase):
    def candidate(self, name, score=90):
        return {"medicine_name": name, "match_score": score}

    def test_direct_brand_and_strength_are_accepted(self):
        result = verify_single_candidate(
            "azilsartan tablets 40 mg abel-40", 0.8,
            self.candidate("abel 40 tablet"),
        )
        self.assertEqual(result["status"], "success")

    def test_variant_suffix_cannot_be_dropped(self):
        result = verify_single_candidate(
            "telmisartan eritel-trio", 0.9,
            self.candidate("eritel 40 tablet"),
        )
        self.assertIn("candidate_variant_conflict", result["reasons"])

    def test_numeric_ocr_cannot_select_strengthless_candidate(self):
        result = verify_single_candidate(
            "clomiphene siphene 100", 0.9,
            self.candidate("siphene tablet"),
        )
        self.assertIn("strengthless_candidate_with_numeric_ocr", result["reasons"])

    def test_unrelated_candidate_is_uncertain(self):
        result = verify_single_candidate(
            "zuventus 132", 0.8, self.candidate("tacloran 1 capsule"),
        )
        self.assertEqual(result["status"], "uncertain")


if __name__ == "__main__":
    unittest.main()
