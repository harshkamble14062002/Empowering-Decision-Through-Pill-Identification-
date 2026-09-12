import unittest
from unittest.mock import patch
from pipeline_multiclass import run_multiclass_pipeline

class PipelineOutcomeTests(unittest.TestCase):
    @patch('pipeline_multiclass.match_medicine')
    @patch('pipeline_multiclass.extract_text_parallel')
    @patch('pipeline_multiclass.detect_multiclass_regions')
    def test_uncertain_never_exposes_final_medicine_details(self,detect,ocr,lookup):
        detect.return_value={'status':'success','detections':{'brand_name':{'crop_path':'unused'}}}
        ocr.return_value={'brand_name':{'status':'success','cleaned_text':'Abel','confidence':.99}}
        lookup.return_value={'status':'uncertain','reasons':['strength_not_read'],'medicine':None,'candidates':[{'medicine_name':'abel 40 tablet'}]}
        result=run_multiclass_pipeline('unused',quiet=True)
        self.assertEqual(result['status'],'uncertain')
        self.assertIsNone(result['final_result'])
        self.assertIn('strength_not_read',result['uncertainty_reasons'])
    @patch('pipeline_multiclass.extract_text_from_image')
    @patch('pipeline_multiclass.match_medicine')
    @patch('pipeline_multiclass.extract_text_parallel')
    @patch('pipeline_multiclass.detect_multiclass_regions')
    def test_expanded_brand_crop_must_pass_normal_matcher(self,detect,ocr,lookup,expanded_ocr):
        detect.return_value={'status':'success','detections':{'brand_name':{
            'crop_path':'tight','expanded_crop_path':'expanded'}}}
        ocr.return_value={'brand_name':{
            'status':'success','cleaned_text':'phylline ab','confidence':.97}}
        expanded_ocr.return_value={
            'status':'success','raw_text':'200 SR Phylline AB',
            'cleaned_text':'200 sr phylline ab','confidence':.90}
        uncertain={'status':'uncertain','reasons':['ambiguous_candidates'],
                   'medicine':None,'candidates':[]}
        confirmed={'status':'success','reasons':[],'candidates':[],
                   'medicine':{'Medicine Name':'ab phylline sr 200 tablet',
                    'Composition':'Acebrophylline (200mg)','Manufacturer':'Sun',
                    'Uses':'use','Side_effects':'effect','match_score':100.0}}
        lookup.side_effect=[uncertain,confirmed]
        result=run_multiclass_pipeline('unused',quiet=True)
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['match_recovery'],'expanded_brand_hints')
        self.assertEqual(result['final_result']['medicine_name'],'ab phylline sr 200 tablet')
    @patch('pipeline_multiclass.match_medicine')
    @patch('pipeline_multiclass.extract_text_parallel')
    @patch('pipeline_multiclass.detect_multiclass_regions')
    def test_empty_brand_cannot_trigger_lookup(self,detect,ocr,lookup):
        detect.return_value={'status':'success','detections':{'brand_name':{'crop_path':'unused'}}}
        ocr.return_value={'brand_name':None}
        result=run_multiclass_pipeline('unused',quiet=True)
        self.assertEqual(result['status'],'uncertain');lookup.assert_not_called()
    @patch('pipeline_multiclass.extract_text_parallel')
    @patch('pipeline_multiclass.detect_multiclass_regions')
    def test_detection_failure_stays_in_denominator(self,detect,ocr):
        detect.return_value={'status':'failed','detections':{}}
        result=run_multiclass_pipeline('unused',quiet=True)
        self.assertEqual(result['status'],'detection_failed');ocr.assert_not_called()

if __name__=='__main__': unittest.main()
