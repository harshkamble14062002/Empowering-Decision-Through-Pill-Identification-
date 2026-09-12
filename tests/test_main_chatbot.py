import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import main_chatbot


class MainChatbotTests(unittest.TestCase):
    def test_uncertain_identification_withholds_answer(self):
        with patch("main_chatbot.run_ocr_csv_pipeline", return_value={
            "status": "uncertain", "final_result": None
        }):
            result = main_chatbot.run(
                "image.jpg", "ಇದರ ಉಪಯೋಗ ಏನು?", index=main_chatbot.DEFAULT_INDEX
            )
        self.assertEqual(result["status"], "identification_uncertain")
        self.assertEqual(result["chatbot"]["status"], "withheld")
        self.assertIsNone(result["chatbot"]["evidence"])

    def test_multiclass_backend_uses_multiclass_pipeline(self):
        identification = {
            "status": "success",
            "final_result": {"medicine_name": "azithral 500 tablet"},
        }
        answer = {"status": "success", "answer": "grounded", "evidence": {}}
        with patch("main_chatbot.run_multiclass_pipeline", return_value=identification) as multiclass, \
             patch("main_chatbot.run_ocr_csv_pipeline") as ocr_csv, \
             patch("main_chatbot.answer_kannada", return_value=answer):
            result = main_chatbot.run(
                "image.jpg", "What is it used for?",
                identification_backend="multiclass",
            )
        multiclass.assert_called_once()
        ocr_csv.assert_not_called()
        self.assertEqual(result["status"], "success")

    def test_confirmed_identification_reaches_grounded_chatbot(self):
        identification = {
            "status": "success",
            "final_result": {"medicine_name": "azithral 500 tablet"},
        }
        answer = {"status": "success", "answer": "grounded", "evidence": {"uses": "evidence"}}
        with patch("main_chatbot.run_ocr_csv_pipeline", return_value=identification), \
             patch("main_chatbot.answer_kannada", return_value=answer) as chatbot:
            result = main_chatbot.run(
                "image.jpg", "ಇದರ ಉಪಯೋಗ ಏನು?", index=main_chatbot.DEFAULT_INDEX
            )
        chatbot.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["chatbot"]["evidence"], {"uses": "evidence"})


if __name__ == "__main__":
    unittest.main()
