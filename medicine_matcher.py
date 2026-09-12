"""Conservative CSV candidate matching with explicit uncertainty; no clinical inference."""
import csv
import re
from functools import lru_cache
from pathlib import Path
from rapidfuzz import fuzz

FORMS = {'tablet','tablets','capsule','capsules','soft','gelatin','mg','mcg','ml','gm'}
IDENTITY_NOISE = {
    'release', 'released', 'sustained', 'extended', 'prolonged', 'modified',
    'coated', 'uncoated', 'film', 'enteric', 'dispersible', 'strip', 'labrets',
}
GENERIC_MANUFACTURER_WORDS = IDENTITY_NOISE | {'tablets', 'capsules', 'tablet', 'capsule'}

def tokens(text):
    return re.findall(r'[a-z]+|\d+(?:\.\d+)?', (text or '').lower())

def key(text):
    return ' '.join(t for t in tokens(text) if t not in FORMS and t not in IDENTITY_NOISE)

def numbers(text):
    return set(re.findall(r'\d+(?:\.\d+)?', text or ''))

def brand(text):
    return ' '.join(
        t for t in tokens(text)
        if t not in FORMS and t not in IDENTITY_NOISE and not t[0].isdigit()
    )

def unordered_ratio(left, right):
    return fuzz.ratio(' '.join(sorted(tokens(left))), ' '.join(sorted(tokens(right))))

@lru_cache(maxsize=4)
def _database(path, modified):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def match_medicine(brand_text, composition_text, manufacturer_text, csv_path,
                   brand_confidence=0.0, manufacturer_confidence=0.0,
                   composition_confidence=0.0):
    """Return candidate-only evidence unless strong, unique identity evidence is present."""
    path=Path(csv_path).resolve()
    rows=_database(str(path),path.stat().st_mtime_ns)
    query=key(brand_text); query_brand=brand(brand_text)
    if not query_brand:
        return {'status':'uncertain','reasons':['missing_brand_text'],'candidates':[],'medicine':None}
    ranked=[]
    for row in rows:
        name=row['Medicine Name']; target_brand=brand(name)
        bscore=max(
            fuzz.ratio(query_brand.replace(' ',''), target_brand.replace(' ','')),
            unordered_ratio(query_brand, target_brand),
        )
        full=max(
            fuzz.ratio(query.replace(' ',''), key(name).replace(' ','')),
            unordered_ratio(query, key(name)),
        )
        # Composition can break close ties, but cannot rescue an unrelated brand.
        comp=fuzz.token_set_ratio(key(composition_text),key(row.get('Composition',''))) if composition_text else 0
        score=.8*bscore+.2*full
        observed_strengths=numbers(brand_text); candidate_strengths=numbers(name)
        if observed_strengths and not observed_strengths.issubset(candidate_strengths):
            score-=30
        elif observed_strengths and candidate_strengths-observed_strengths:
            score-=10
        if bscore>=85 and comp>=75: score=min(100,score+2)
        ranked.append((score,bscore,full,comp,row))
    ranked.sort(key=lambda x:x[0],reverse=True)
    best=ranked[0]
    resolution = None

    # Composition may resolve a close brand family only when it strongly and uniquely
    # identifies a multi-ingredient variant. It cannot rescue an unrelated brand.
    plausible = [item for item in ranked if item[1] >= 80]
    composition_ranked = sorted(plausible, key=lambda item: item[3], reverse=True)
    if composition_confidence >= .70 and composition_text and composition_ranked:
        observed_query_strengths = numbers(brand_text)
        strength_composition_matches = [
            item for item in plausible
            if observed_query_strengths
            and observed_query_strengths.issubset(numbers(item[4]['Medicine Name']))
            and item[3] >= 75
        ]
        if len(strength_composition_matches) == 1:
            best = strength_composition_matches[0]
            resolution = 'strength_composition_tiebreak'
        else:
            composition_best = composition_ranked[0]
            next_composition = composition_ranked[1][3] if len(composition_ranked) > 1 else 0
            ingredient_words = {
                token for token in tokens(composition_best[4].get('Composition', ''))
                if len(token) > 4 and token not in FORMS
                and token not in {'hydrochloride', 'sodium', 'potassium'}
            }
            if (composition_best[3] >= 78 and composition_best[3] - next_composition >= 15
                    and len(ingredient_words) >= 2):
                best = composition_best
                resolution = 'composition_tiebreak'

    score,bscore,full,comp,row=best
    ordered = [best] + [item for item in ranked if item is not best]
    candidates=[{'medicine_name':r['Medicine Name'],'composition':r.get('Composition',''),'manufacturer':r.get('Manufacturer',''),'score':round(s,2)} for s,b,f,c,r in ordered[:3]]
    if bscore<60:
        return {'status':'no_match','reasons':['no_plausible_brand'],'candidates':[], 'medicine':None}
    reasons=[]
    if brand_confidence<.70: reasons.append('low_brand_ocr_confidence')
    if not resolution and (bscore<92 or full<85): reasons.append('weak_brand_identity')
    if not resolution and len(ranked)>1 and score-ranked[1][0]<12: reasons.append('ambiguous_candidates')
    observed=numbers(brand_text); expected=numbers(row['Medicine Name'])
    if observed and not observed.issubset(expected): reasons.append('strength_conflict')
    # Never guess a strength variant when package OCR omits all its numbers.
    if expected and not observed: reasons.append('strength_not_read')
    # Compare explicit unit-bearing composition amounts without inventing conversions.
    amounts=lambda text: set(re.findall(r'(\d+(?:\.\d+)?)\s*(mg|mcg|ml)\b', (text or '').lower()))
    observed_amounts=amounts(composition_text); expected_amounts=amounts(row.get('Composition',''))
    if observed_amounts and expected_amounts and not observed_amounts.issubset(expected_amounts):
        reasons.append('composition_strength_conflict')
    manufacturer_words = [
        token for token in tokens(manufacturer_text)
        if len(token) >= 4 and token not in FORMS and token not in GENERIC_MANUFACTURER_WORDS
    ]
    if manufacturer_words and manufacturer_confidence>=.70:
        observed_manufacturer = ' '.join(manufacturer_words)
        if fuzz.token_set_ratio(observed_manufacturer,brand(row.get('Manufacturer','')))<60:
            reasons.append('manufacturer_conflict')
    comp_words={t for t in tokens(composition_text) if len(t)>4 and t not in FORMS and t not in {'hydrochloride','tablets','capsules','release','sustained','extended','prolonged','modified'}}
    target_words={t for t in tokens(row.get('Composition','')) if len(t)>4}
    if comp_words and target_words and max(fuzz.ratio(a,b) for a in comp_words for b in target_words)<65:
        reasons.append('composition_conflict')
    status='uncertain' if reasons else 'success'
    return {'status':status,'reasons':reasons,'candidates':candidates,
            'resolution':resolution,
            'medicine':dict(row,match_score=round(score,2)) if status=='success' else None}
