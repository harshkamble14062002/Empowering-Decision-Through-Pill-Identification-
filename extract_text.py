#!/usr/bin/env python3
"""
extract_text.py - EasyOCR Text Extraction
Sprint 3, Task 2: Extract medicine name text from cropped image regions
"""

import argparse
import os
import sys
import re
import cv2
import numpy as np
import easyocr
import threading

_READERS = {}
_READER_LOCK = threading.RLock()

def _read_ocr(image, lang):
    # Share model initialization and serialize access to the GPU reader.
    with _READER_LOCK:
        key = tuple(lang)
        if key not in _READERS:
            import torch
            try:
                _READERS[key] = easyocr.Reader(list(lang), gpu=torch.cuda.is_available())
            except Exception:
                _READERS[key] = easyocr.Reader(list(lang), gpu=False)
        return _READERS[key].readtext(image)


def extract_text_from_image(image_path, lang=['en'], conf_threshold=0.5, preprocess=True):
    """
    Extract text from medicine image using EasyOCR
    
    Args:
        image_path: Path to input image (preferably cropped region)
        lang: List of languages for OCR (default: ['en'])
        conf_threshold: Minimum confidence for text results
        preprocess: Whether to apply preprocessing
    
    Returns:
        dict: Extracted text with confidence and metadata
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Preprocess if enabled
    if preprocess:
        print("Applying preprocessing...")
        image = preprocess_for_ocr(image)
    
    results = _read_ocr(image, lang)

    if not results:
        print("⚠️  No text detected in image")
        return {
            'raw_text': '',
            'cleaned_text': '',
            'confidence': 0.0,
            'all_detections': [],
            'status': 'no_text'
        }
    
    # Process results
    all_detections = []
    for bbox, text, conf in results:
        if conf >= conf_threshold:
            all_detections.append({
                'text': text,
                'confidence': float(conf),
                'bbox': bbox,
                'y_center': sum(p[1] for p in bbox) / 4
            })
    
    if not all_detections:
        print(f"⚠️  No text with confidence >= {conf_threshold}")
        return {
            'raw_text': '',
            'cleaned_text': '',
            'confidence': 0.0,
            'all_detections': [],
            'status': 'low_confidence'
        }
    
    # Combine all detected text (sorted from top to bottom roughly)
    all_detections.sort(key=lambda x: x['y_center'])
    raw_text = ' '.join([d['text'] for d in all_detections])
    avg_confidence = sum([d['confidence'] for d in all_detections]) / len(all_detections)
    
    # Clean text
    cleaned_text = clean_medicine_text(raw_text)
    
    print(f"✅ Text extracted:")
    print(f"   Raw text: {raw_text}")
    print(f"   Cleaned text: {cleaned_text}")
    print(f"   Average confidence: {avg_confidence:.2%}")
    print(f"   Number of text regions: {len(all_detections)}")
    
    return {
        'raw_text': raw_text,
        'cleaned_text': cleaned_text,
        'confidence': avg_confidence,
        'all_detections': all_detections,
        'status': 'success'
    }


def preprocess_for_ocr(image):
    """
    Apply preprocessing pipeline to improve OCR accuracy
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        processed_image: Enhanced image for OCR
    """
    # 1. Resize image (make it larger so OCR can read small text better)
    scale_factor = 2.0
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 4. Sharpen (more aggressive)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to BGR for EasyOCR
    processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return processed


def clean_medicine_text(text):
    """
    Clean extracted text for medicine name matching
    
    Args:
        text: Raw text from OCR
    
    Returns:
        cleaned_text: Processed text ready for database lookup
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove common OCR artifacts
    # Keep only alphanumeric, spaces, dots, and hyphens
    text = re.sub(r'[^a-z0-9\s.\-]', '', text)
    
    # Normalize multiple dots/hyphens
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\-+', '-', text)
    
    return text


def main():
    parser = argparse.ArgumentParser(
        description='Extract medicine name text from image using EasyOCR'
    )
    parser.add_argument('--image', '-i', required=True,
                        help='Path to input image (cropped region)')
    parser.add_argument('--lang', '-l', nargs='+', default=['en'],
                        help='Language(s) for OCR (default: en)')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Disable image preprocessing')
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        result = extract_text_from_image(
            image_path=args.image,
            lang=args.lang,
            conf_threshold=args.conf,
            preprocess=not args.no_preprocess
        )
        
        print("\n" + "="*60)
        print("OCR Summary:")
        print("="*60)
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Cleaned text: {result['cleaned_text']}")
            print(f"Average confidence: {result['confidence']:.2%}")
            print(f"Text regions detected: {len(result['all_detections'])}")
        
        # Save to JSON if requested
        if args.output:
            import json
            
            # Helper function to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def extract_text_parallel(detections, conf_threshold=0.4):
    """
    Extract text from multiple crops in parallel
    
    Args:
        detections: Dict of {class_name: detection_dict} from detect_multiclass_regions
        conf_threshold: Minimum OCR confidence threshold
    
    Returns:
        dict: {class_name: ocr_result} for each detection
    """
    from concurrent.futures import ThreadPoolExecutor
    
    def ocr_single(class_name, detection):
        if detection is None:
            return class_name, None
        try:
            return class_name, extract_text_from_image(
                detection['crop_path'], 
                lang=['en'],
                conf_threshold=conf_threshold,
                preprocess=True
            )
        except Exception as e:
            return class_name, {'status': 'error', 'error': str(e)}
    
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(ocr_single, name, det) 
                   for name, det in detections.items()]
        for future in futures:
            name, result = future.result()
            results[name] = result
    
    return results


if __name__ == '__main__':
    sys.exit(main())
