import unittest

from single_class_safety import verify_single_candidate


class SingleClassSafetyTests(unittest.TestCase):
    def candidate(self, name, score=90):
        return {"medicine_name": name, "match_score": score}

    def test_direct_brand_and_strength_are_accepted(self):
        result = verify_single_candidate(
            "azilsartan tablets 40 mg abel-40", 0.8,
            self.candidate("abel 40 tablet"),
        )
        self.assertEqual(result["status"], "success")

    def test_unread_brand_is_uncertain(self):
        result = verify_single_candidate(
            "zuventus 132", 0.8, self.candidate("tacloran 1 capsule"),
        )
        self.assertEqual(result["status"], "uncertain")

    def test_unread_strength_is_uncertain(self):
        result = verify_single_candidate(
            "abel", 0.9, self.candidate("abel 40 tablet"),
        )
        self.assertIn("candidate_strength_not_read", result["reasons"])

    def test_low_match_score_is_uncertain(self):
        result = verify_single_candidate(
            "abel 40", 0.9, self.candidate("abel 40 tablet", 50),
        )
        self.assertIn("low_lookup_score", result["reasons"])

    def test_no_candidate_is_no_match(self):
        self.assertEqual(
            verify_single_candidate("anything", 0.9, None)["status"],
            "no_match",
        )


if __name__ == "__main__":
    unittest.main()
