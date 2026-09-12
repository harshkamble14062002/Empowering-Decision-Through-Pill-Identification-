#!/usr/bin/env python3
"""
detect_medicine_multiclass.py - YOLOv5 Multi-Class Medicine Detection
Detects brand_name, composition, and manufacturer regions separately
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import math
import numpy as np
import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=2)
def _load_model(model_path, modified):
    cached_repo = Path(torch.hub.get_dir()) / 'ultralytics_yolov5_master'
    if cached_repo.is_dir():
        model = torch.hub.load(str(cached_repo), 'custom', path=model_path, source='local')
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    # FP16 validation was unreliable on the local GTX 1650; keep inference FP32.
    model.float().eval()
    model.amp = False
    return model


def detect_multiclass_regions(image_path, model_path, output_dir='output', conf_threshold=0.25):
    """
    Detect brand_name, composition, and manufacturer regions using multi-class YOLOv5 model
    
    Args:
        image_path: Path to input medicine image
        model_path: Path to trained multi-class YOLOv5 model
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections (0-1)
    
    Returns:
        dict: Detection results with status and per-class best detections
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    cropped_dir = os.path.join(output_dir, 'cropped', 'multiclass')
    annotated_dir = os.path.join(output_dir, 'annotated')
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    
    print(f"Loading multi-class YOLOv5 model from: {model_path}")
    model_file = Path(model_path).resolve()
    model = _load_model(str(model_file), model_file.stat().st_mtime_ns)
    model.conf = conf_threshold
    
    print(f"Running inference on: {image_path}")
    # Match validation's resize-then-pad preprocessing rather than AutoShape.
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    ratio = 768 / max(height, width)
    resized = cv2.resize(
        image, (math.ceil(width * ratio), math.ceil(height * ratio)),
        interpolation=cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA,
    )
    stride = int(model.stride)
    padded_size = math.ceil(768 / stride + 0.5) * stride
    padded, _, (pad_x, pad_y) = letterbox(
        resized, (padded_size, padded_size), auto=False, scaleup=False,
    )
    parameter = next(model.parameters())
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)[::-1]))
    tensor = tensor.to(parameter.device).float().unsqueeze(0) / 255
    with torch.inference_mode():
        predictions = model.model(tensor)
        boxes = non_max_suppression(predictions, conf_threshold, 0.6)[0]
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / (resized.shape[1] / width)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / (resized.shape[0] / height)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    detections_df = pd.DataFrame(
        boxes.cpu().numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'],
    )
    
    if len(detections_df) == 0:
        print("⚠️  No regions detected")
        return {'status': 'failed', 'detections': {}}
    
    class_names = {0: 'brand_name', 1: 'composition', 2: 'manufacturer'}
    best_detections = {}
    
    image = cv2.imread(image_path)
    h_img, w_img = image.shape[:2]
    image_name = Path(image_path).stem
    
    annotated_image = image.copy()
    colors = {0: (0, 255, 0), 1: (255, 165, 0), 2: (0, 0, 255)}
    
    for class_id, group in detections_df.groupby('class'):
        best = group.loc[group['confidence'].idxmax()]
        class_name = class_names[int(class_id)]
        
        x1, y1, x2, y2 = int(best.xmin), int(best.ymin), int(best.xmax), int(best.ymax)
        w, h = x2 - x1, y2 - y1
        
        pad_w, pad_h = int(w * 0.1), int(h * 0.1)
        
        crop_y1 = max(0, y1 - pad_h)
        crop_y2 = min(h_img, y2 + pad_h)
        crop_x1 = max(0, x1 - pad_w)
        crop_x2 = min(w_img, x2 + pad_w)
        
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_path = os.path.join(cropped_dir, f"{image_name}_{class_name}.jpg")
        cv2.imwrite(crop_path, crop)

        detection = {
            'bbox': {'x': x1, 'y': y1, 'w': w, 'h': h},
            'confidence': float(best.confidence),
            'crop_path': crop_path
        }
        # Keep the precise detector crop as the first OCR source. An expanded brand
        # crop is available only as a guarded retry when the precise text is ambiguous.
        if class_name == 'brand_name':
            expanded_y1 = max(0, y1 - h)
            expanded_y2 = min(h_img, y2 + h)
            expanded_x1 = max(0, x1 - w)
            expanded_x2 = min(w_img, x2 + w)
            expanded = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            expanded_path = os.path.join(
                cropped_dir, f"{image_name}_{class_name}_expanded.jpg"
            )
            cv2.imwrite(expanded_path, expanded)
            detection['expanded_crop_path'] = expanded_path

        best_detections[class_name] = detection
        
        color = colors.get(int(class_id), (255, 255, 255))
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {best.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        print(f"✅ {class_name}: conf={best.confidence:.2%}, bbox=({x1},{y1},{w},{h})")
    
    annotated_path = os.path.join(annotated_dir, f"{image_name}_multiclass.jpg")
    cv2.imwrite(annotated_path, annotated_image)
    
    status = 'success' if 'brand_name' in best_detections else 'partial'
    return {
        'status': status,
        'detections': best_detections,
        'annotated_path': annotated_path,
        'image_path': image_path
    }


def main():
    parser = argparse.ArgumentParser(
        description='Detect medicine regions (brand_name, composition, manufacturer) using multi-class YOLOv5'
    )
    parser.add_argument('--image', '-i', required=True,
                        help='Path to input medicine image')
    parser.add_argument('--model', '-m', default='multiclass_models/best_multiclass.pt',
                        help='Path to multi-class YOLOv5 model')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory for results (default: output)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    try:
        result = detect_multiclass_regions(
            image_path=args.image,
            model_path=args.model,
            output_dir=args.output,
            conf_threshold=args.conf
        )
        
        print("\n" + "="*60)
        print("Multi-Class Detection Summary:")
        print("="*60)
        print(f"Status: {result['status']}")
        for class_name, detection in result['detections'].items():
            print(f"{class_name}: {detection['confidence']:.2%} - {detection['crop_path']}")
        
        return 0 if result['status'] in ['success', 'partial'] else 1
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
