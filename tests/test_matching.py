import unittest

from app.ocr import brand_evidence, is_irrelevant, preclassify_lines, rank_rows


class OcrCsvClassifierTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "Medicine Name": "acemiz 200 sr tablet",
                "Composition": "Aceclofenac (200mg)",
                "Manufacturer": "Lupin Ltd",
            },
            {
                "Medicine Name": "lupin acp 100mg/325mg tablet",
                "Composition": "Aceclofenac (100mg) + Paracetamol (325mg)",
                "Manufacturer": "Lupin Ltd",
            },
            {
                "Medicine Name": "topnac 200mg tablet sr",
                "Composition": "Aceclofenac (200mg)",
                "Manufacturer": "Other Labs",
            },
        ]
        texts = ["10 x 10 Tablets", "Aceclofenac Sustained", "Aceriz 200 SR", "LUPIN"]
        self.lines = [
            {
                "text": text,
                "confidence": 0.9,
                "bbox": [0.0, float(i), 10.0, float(i + 1)],
            }
            for i, text in enumerate(texts)
        ]

    def test_packaging_text_is_ignored(self):
        self.assertTrue(is_irrelevant("10 x 10 Tablets"))
        self.assertTrue(is_irrelevant("Store below 25 C"))
        self.assertFalse(is_irrelevant("Aceriz 200 SR"))

    def test_brand_evidence_rejects_strength_only_and_composition_text(self):
        self.assertGreaterEqual(
            brand_evidence({"text": "Sonaxa 75"}, "sonaxa 75 capsule"), 99
        )
        self.assertEqual(
            brand_evidence({"text": "75"}, "renerve p 750mcg/75mg capsule"), 0
        )
        self.assertEqual(
            brand_evidence(
                {"text": "Pregabalin"}, "genericart pregabalin 75mg capsule"
            ),
            0,
        )

    def test_manufacturer_cannot_compete_as_brand(self):
        classified = preclassify_lines(self.lines, self.rows)
        by_text = {item["text"]: item["preliminary_class"] for item in classified}
        self.assertEqual(by_text["LUPIN"], "manufacturer")
        self.assertEqual(by_text["Aceriz 200 SR"], "brand_name")
        self.assertEqual(by_text["Aceclofenac Sustained"], "composition")
        self.assertEqual(
            rank_rows(classified, self.rows)[0]["row"]["Medicine Name"],
            "acemiz 200 sr tablet",
        )


if __name__ == "__main__":
    unittest.main()
