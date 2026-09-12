import unittest

from openfda_details import answer_from_openfda, ingredients_from_composition, label_intent


class OpenFdaDetailsTests(unittest.TestCase):
    def test_combination_ingredients_are_parsed(self):
        self.assertEqual(
            ingredients_from_composition("Amoxycillin (500mg) + Clavulanic Acid (125mg)"),
            ["Amoxycillin", "Clavulanic Acid"],
        )

    def test_kannada_dosage_intent(self):
        self.assertEqual(label_intent("ಡೋಸ್ ಎಷ್ಟು ತೆಗೆದುಕೊಳ್ಳಬೇಕು?"), "dosage")

    def test_answer_preserves_source_and_jurisdiction(self):
        details = {
            "status": "success",
            "label": {"dosage": "Label directions", "uses": ""},
            "jurisdiction": "United States FDA product label",
            "effective_time": "20250101",
            "source_url": "https://example.test/label",
        }
        result = answer_from_openfda(details, "ಡೋಸ್ ಎಷ್ಟು ತೆಗೆದುಕೊಳ್ಳಬೇಕು?")
        self.assertEqual(result["evidence"], "Label directions")
        self.assertEqual(result["jurisdiction"], "United States FDA product label")
        self.assertEqual(result["source_url"], "https://example.test/label")


if __name__ == "__main__":
    unittest.main()
