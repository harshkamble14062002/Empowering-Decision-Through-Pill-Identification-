import csv,tempfile,unittest
from pathlib import Path
from medicine_matcher import match_medicine

class MedicineIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.path=Path(self.temp.name)/'medicines.csv'
        with self.path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['Medicine Name','Composition','Manufacturer']);w.writeheader()
            w.writerows([
                {'Medicine Name':'abel 40 tablet','Composition':'Azilsartan (40mg)','Manufacturer':'Lupin Ltd'},
                {'Medicine Name':'abel 80 tablet','Composition':'Azilsartan (80mg)','Manufacturer':'Lupin Ltd'},
                {'Medicine Name':'aa 5 tablet','Composition':'Levocetirizine (5mg)','Manufacturer':'Blaze Remedies'},
                {'Medicine Name':'olan 5 tablet','Composition':'Olanzapine (5mg)','Manufacturer':'Other Labs'},
                {'Medicine Name':'zoryl m 2 tablet','Composition':'Glimepiride (2mg) + Metformin (500mg)','Manufacturer':'Intas'},
                {'Medicine Name':'zoryl m 2 forte tablet','Composition':'Glimepiride (2mg) + Metformin (1000mg)','Manufacturer':'Intas'},
                {'Medicine Name':'ab phylline capsule','Composition':'Acebrophylline (100mg)','Manufacturer':'Sun Pharmaceutical Industries Ltd'},
                {'Medicine Name':'ab phylline n tablet','Composition':'Acebrophylline (100mg) + Acetylcysteine (600mg)','Manufacturer':'Sun Pharmaceutical Industries Ltd'},
                {'Medicine Name':'ab phylline sr 200 tablet','Composition':'Acebrophylline (200mg)','Manufacturer':'Sun Pharmaceutical Industries Ltd'},
                {'Medicine Name':'acemiz 100mg tablet','Composition':'Aceclofenac (100mg)','Manufacturer':'Lupin Ltd'},
                {'Medicine Name':'acemiz 200 sr tablet','Composition':'Aceclofenac (200mg)','Manufacturer':'Lupin Ltd'},
                {'Medicine Name':'acemiz plus tablet','Composition':'Aceclofenac (100mg) + Paracetamol (325mg)','Manufacturer':'Lupin Ltd'},
            ])
    def tearDown(self): self.temp.cleanup()
    def match(self,text,comp='',mfg='',**kwargs):
        return match_medicine(
            text,comp,mfg,self.path,
            brand_confidence=kwargs.get('brand_confidence',.99),
            manufacturer_confidence=.99,
            composition_confidence=kwargs.get('composition_confidence',0),
        )
    def test_strength_variants_are_not_guessed(self):
        self.assertEqual(self.match('Abel')['status'],'uncertain')
    def test_visible_strength_identifies_variant(self):
        result=self.match('Abel-40')
        self.assertEqual(result['status'],'success');self.assertEqual(result['medicine']['Medicine Name'],'abel 40 tablet')
    def test_unknown_strength_is_not_accepted(self):
        self.assertEqual(self.match('Abel-20')['status'],'uncertain')
    def test_manufacturer_conflict_is_uncertain(self):
        self.assertEqual(self.match('Abel-40',mfg='Cipla')['status'],'uncertain')
    def test_unrelated_composition_is_uncertain(self):
        self.assertEqual(self.match('Abel-40',comp='Paracetamol')['status'],'uncertain')
    def test_composition_strength_conflict(self):
        self.assertEqual(self.match('Abel-40',comp='Azilsartan 80mg')['status'],'uncertain')
    def test_empty_and_absent_medicine(self):
        self.assertEqual(self.match('')['status'],'uncertain')
        self.assertEqual(self.match('Qzxwvv')['status'],'no_match')
    def test_aa5_regression(self):
        self.assertEqual(self.match('AA-5')['medicine']['Medicine Name'],'aa 5 tablet')
        self.assertNotEqual(self.match('blaze 5m3')['status'],'success')
    def test_strength_and_composition_resolve_single_ingredient_variant(self):
        result=self.match(
            'Acemiz 200', comp='Aceclofenac Sustained Release Tablets',
            composition_confidence=.92,
        )
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['resolution'],'strength_composition_tiebreak')
        self.assertEqual(result['medicine']['Medicine Name'],'acemiz 200 sr tablet')
    def test_composition_without_strength_does_not_guess_acemiz_variant(self):
        result=self.match(
            'Acemiz', comp='Aceclofenac Sustained Release Tablets',
            composition_confidence=.92,
        )
        self.assertEqual(result['status'],'uncertain')
    def test_unique_high_confidence_composition_resolves_n_variant(self):
        result=self.match(
            'ABPbylline',
            comp='Acebrophyllige and Acetylcysteine Tablets',
            composition_confidence=.85,
        )
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['resolution'],'composition_tiebreak')
        self.assertEqual(result['medicine']['Medicine Name'],'ab phylline n tablet')
    def test_low_confidence_composition_does_not_resolve_variant(self):
        result=self.match(
            'ABPbylline',
            comp='Acebrophyllige and Acetylcysteine Tablets',
            composition_confidence=.5,
        )
        self.assertEqual(result['status'],'uncertain')
    def test_reordered_brand_with_strength_is_unique(self):
        result=self.match('200 SR Phylline AB',mfg='ed Release Tablets')
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['medicine']['Medicine Name'],'ab phylline sr 200 tablet')
    def test_partial_reordered_brand_remains_uncertain(self):
        self.assertEqual(self.match('Phylline AB')['status'],'uncertain')
    def test_low_confidence_ocr_is_not_accepted(self):
        self.assertEqual(self.match('Abel-40',brand_confidence=.4)['status'],'uncertain')
    def test_forte_is_preserved(self):
        self.assertEqual(self.match('Zoryl M 2 Forte')['medicine']['Medicine Name'],'zoryl m 2 forte tablet')

if __name__=='__main__': unittest.main()
