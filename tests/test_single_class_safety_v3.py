import unittest

from single_class_safety_v3 import verify_single_candidate


class SingleClassSafetyV3Tests(unittest.TestCase):
    def test_joined_variant_suffix_is_uncertain(self):
        candidate = {"medicine_name": "telma 80 tablet", "match_score": 90}
        result = verify_single_candidate(
            "telmisartan 80mg telmact glenmark", 0.9, candidate
        )
        self.assertIn("joined_candidate_variant_conflict", result["reasons"])

    def test_matching_brand_and_strength_remain_accepted(self):
        candidate = {"medicine_name": "abel 40 tablet", "match_score": 90}
        result = verify_single_candidate("azilsartan 40mg abel-40", 0.9, candidate)
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
