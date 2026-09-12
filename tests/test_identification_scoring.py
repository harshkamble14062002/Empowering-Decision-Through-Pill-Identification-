import unittest
from identification_evaluation.scoring import identity_outcome

class IdentificationScoringTests(unittest.TestCase):
    def test_success_status_does_not_make_wrong_strength_correct(self):
        r={'status':'success','final_result':{'medicine_name':'abel 80 tablet'}}
        self.assertEqual(identity_outcome(r,'abel 40 tablet'),'wrong')
    def test_missing_name_in_success_is_wrong(self):
        self.assertEqual(identity_outcome({'status':'success'},'abel 40 tablet'),'wrong')
    def test_uncertain_candidate_is_not_correct_identification(self):
        r={'status':'uncertain','final_result':{'medicine_name':'abel 40 tablet'}}
        self.assertEqual(identity_outcome(r,'abel 40 tablet'),'uncertain')
    def test_case_and_whitespace_do_not_change_identity(self):
        r={'status':'success','final_result':{'medicine_name':' ABEL  40 tablet '}}
        self.assertEqual(identity_outcome(r,'abel 40 tablet'),'correct')
