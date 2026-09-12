#!/usr/bin/env python3
"""
pipeline_multiclass.py - Multi-Class Medicine Detection Pipeline
End-to-end pipeline using multi-class detection with parallel OCR
"""

import argparse
import time
import json
import sys
import re
from pathlib import Path
from detect_medicine_multiclass import detect_multiclass_regions
from extract_text import extract_text_from_image, extract_text_parallel
from medicine_matcher import match_medicine

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = str(PROJECT_ROOT / "multiclass_models/best_multiclass.pt")
DEFAULT_DATABASE = str(PROJECT_ROOT / "Medicine_Details_Verified.csv")


def run_multiclass_pipeline(image_path, model_path=DEFAULT_MODEL, database_path=DEFAULT_DATABASE, 
                            output_dir='output', quiet=False):
    """
    Run complete multi-class medicine identification pipeline
    
    Args:
        image_path: Path to medicine image
        model_path: Path to multi-class YOLOv5 model
        database_path: Path to medicine database CSV
        output_dir: Output directory for crops and annotations
        quiet: Suppress progress output
    
    Returns:
        dict: Pipeline result with status, timings, and match data
    """
    start = time.time()
    result = {'image_path': str(image_path), 'model_path': str(model_path),
              'inference_precision': 'float32', 'image_size': 768,
              'preprocessing': 'resize longest side to 768, then square pad with half-stride margin',
              'status': 'pending', 'timings': {}, 'stages': {}}
    
    t0 = time.time()
    det_result = detect_multiclass_regions(image_path, model_path, output_dir)
    result['timings']['detection'] = time.time() - t0
    result['stages']['detection'] = det_result
    
    if det_result['status'] == 'failed':
        result['status'] = 'detection_failed'
        result['total_time'] = time.time() - start
        return result
    
    detections = det_result['detections']
    if 'brand_name' not in detections:
        result['status'] = 'no_brand_detected'
        result['total_time'] = time.time() - start
        return result
    
    t0 = time.time()
    ocr_results = extract_text_parallel(detections)
    result['timings']['ocr'] = time.time() - t0
    result['stages']['ocr'] = {
        name: {key: value.get(key) for key in
               ('status', 'raw_text', 'cleaned_text', 'confidence', 'error')
               if key in value}
        for name, value in ocr_results.items() if value is not None
    }
    
    brand_text = (ocr_results.get('brand_name') or {}).get('cleaned_text', '')
    comp_text = ocr_results.get('composition', {}).get('cleaned_text') if ocr_results.get('composition') else None
    mfg_text = ocr_results.get('manufacturer', {}).get('cleaned_text') if ocr_results.get('manufacturer') else None
    
    if not brand_text or len(brand_text) < 2:
        result['status'] = 'uncertain'
        result['uncertainty_reasons'] = ['missing_brand_text']
        result['final_result'] = None
        result['total_time'] = time.time() - start
        return result
    
    t0 = time.time()
    decision = match_medicine(
        brand_text, comp_text, mfg_text, database_path,
        brand_confidence=(ocr_results.get('brand_name') or {}).get('confidence', 0),
        manufacturer_confidence=(ocr_results.get('manufacturer') or {}).get('confidence', 0),
        composition_confidence=(ocr_results.get('composition') or {}).get('confidence', 0),
    )

    # A tight detector crop can omit an adjacent strength or suffix. Retry once with
    # the detector-anchored expanded crop, but accept it only if the normal matcher
    # reaches an unambiguous success decision.
    expanded_path = detections['brand_name'].get('expanded_crop_path')
    if decision['status'] != 'success' and expanded_path:
        recovery_start = time.time()
        expanded_ocr = extract_text_from_image(
            expanded_path, lang=['en'], conf_threshold=0.3, preprocess=True
        )
        result['timings']['ocr_recovery'] = time.time() - recovery_start
        result['stages']['ocr']['brand_name_expanded'] = {
            key: expanded_ocr.get(key) for key in
            ('status', 'raw_text', 'cleaned_text', 'confidence', 'error')
            if key in expanded_ocr
        }
        expanded_text = expanded_ocr.get('cleaned_text', '')
        if expanded_text and expanded_text != brand_text:
            # Preserve the reliable tight-crop brand and add only identity-bearing
            # strength/formulation hints from the wider crop. This avoids treating
            # nearby composition or package-count text as part of the brand.
            hint_tokens = re.findall(r'\d+(?:\.\d+)?|\b(?:sr|cr|xr|er|mr|od|n|plus|forte)\b', expanded_text)
            hint_text = ' '.join(dict.fromkeys(hint_tokens))
            recovered = None
            if hint_text and expanded_ocr.get('confidence', 0) >= .60:
                recovered = match_medicine(
                    f'{brand_text} {hint_text}', comp_text, mfg_text, database_path,
                    brand_confidence=(ocr_results.get('brand_name') or {}).get('confidence', 0),
                    manufacturer_confidence=(ocr_results.get('manufacturer') or {}).get('confidence', 0),
                    composition_confidence=(ocr_results.get('composition') or {}).get('confidence', 0),
                )
                if recovered['status'] == 'success':
                    decision = recovered
                    result['match_recovery'] = 'expanded_brand_hints'
            if decision['status'] != 'success':
                recovered = match_medicine(
                    expanded_text, comp_text, mfg_text, database_path,
                    brand_confidence=expanded_ocr.get('confidence', 0),
                    manufacturer_confidence=(ocr_results.get('manufacturer') or {}).get('confidence', 0),
                    composition_confidence=(ocr_results.get('composition') or {}).get('confidence', 0),
                )
                if recovered['status'] == 'success':
                    decision = recovered
                    result['match_recovery'] = 'expanded_brand_crop'

    lookup_result = decision['medicine']
    strategy = decision.get('resolution') or 'verified_brand_candidate'
    result['stages']['lookup'] = decision
    result['candidates'] = decision['candidates']
    result['uncertainty_reasons'] = decision['reasons']
    result['timings']['lookup'] = time.time() - t0
    result['match_strategy'] = strategy
    
    if not lookup_result:
        result['status'] = decision['status']
        result['final_result'] = None
        if not quiet:
            print(f"{decision['status']}: {', '.join(decision['reasons'])}")
        result['total_time'] = time.time() - start
        return result
    
    result['status'] = 'success'
    result['final_result'] = {
        'medicine_name': lookup_result.get('Medicine Name'),
        'composition': lookup_result.get('Composition'),
        'manufacturer': lookup_result.get('Manufacturer'),
        'uses': lookup_result.get('Uses'),
        'side_effects': lookup_result.get('Side_effects'),
        'match_score': lookup_result.get('match_score'),
        'match_strategy': strategy
    }
    result['total_time'] = time.time() - start
    
    if not quiet:
        print(f"✅ {result['final_result']['medicine_name']} ({strategy}, {result['final_result']['match_score']:.1f}%) - {result['total_time']:.2f}s")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Multi-class medicine detection pipeline'
    )
    parser.add_argument('--image', required=True,
                        help='Path to medicine image')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Path to multi-class model (default: multiclass_models/best_multiclass.pt)')
    parser.add_argument('--database', default=DEFAULT_DATABASE,
                        help='Path to medicine database (default: Medicine_Details_Verified.csv)')
    parser.add_argument('--output', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--save-json',
                        help='Save result to JSON file')
    
    args = parser.parse_args()
    
    result = run_multiclass_pipeline(
        args.image, 
        args.model, 
        args.database, 
        args.output, 
        args.quiet
    )
    
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(result, f, indent=2)
    
    sys.exit(0 if result['status'] == 'success' else 2 if result['status'] == 'uncertain' else 1)


if __name__ == '__main__':
    main()
