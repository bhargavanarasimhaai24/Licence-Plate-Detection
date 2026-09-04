"""
==============================================================================
SIH 2026 - MODEL 1: VEHICLE DETECTION & CROPPING PIPELINE (YOLO11n)
==============================================================================
This module performs vehicle detection on input images using YOLO11n,
extracts individual vehicle cropped images for downstream ANPR & LP detection
models, and outputs structured JSON detection metadata.

Target Classes (5):
  0: auto_rickshaw
  1: car
  2: bus
  3: motorcycle
  4: truck
==============================================================================
"""

import os
import sys
import json
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

# Target SIH 2026 class dictionary
TARGET_CLASS_NAMES = {
    0: "auto_rickshaw",
    1: "car",
    2: "bus",
    3: "motorcycle",
    4: "truck"
}

# COCO vehicle class index to SIH 5-class mapping
COCO_TO_SIH_MAPPING = {
    2: (1, "car"),
    3: (3, "motorcycle"),
    5: (2, "bus"),
    7: (4, "truck")
}

def detect_vehicles(
    image_path: str,
    model_path: str = "yolo11n.pt",
    output_dir: str = "output",
    conf_thresh: float = 0.25,
    imgsz: int = 800
) -> dict:
    """
    Perform vehicle detection on an input image.

    Args:
        image_path (str): Path to input image file.
        model_path (str): Path to YOLO model weights file.
        output_dir (str): Root output directory.
        conf_thresh (float): Confidence threshold for detections.
        imgsz (int): Inference image resolution.

    Returns:
        dict: Detection metadata structure containing vehicle bounding boxes,
              classes, confidence scores, and crop file paths.
    """
    image_path = Path(image_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found at: {image_path.resolve()}")

    # Check custom model vs COCO model
    custom_weights = Path("models/vehicle_detector_yolo11n.pt")
    if model_path.name == "vehicle_detector_yolo11n.pt" and not custom_weights.exists():
        print(f"[WARNING] Custom weights '{custom_weights}' not found. Using 'yolo11n.pt'...")
        model_path = Path("yolo11n.pt")

    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("MODEL 1: VEHICLE DETECTION INFERENCE")
    print(f"{'='*60}")
    print(f"Loading Model Weights : {model_path}")
    print(f"Input Image Path      : {image_path.resolve()}")
    print(f"Confidence Threshold  : {conf_thresh}")
    print(f"Inference Image Size  : {imgsz}")

    # Load YOLO Model
    model = YOLO(str(model_path))
    
    # Determine if using COCO weights (needs class filtering and remapping)
    is_coco_model = "car" in model.names.values() and len(model.names) > 10

    if is_coco_model:
        # Filter for vehicle classes in COCO: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        results = model.predict(
            source=str(image_path),
            conf=conf_thresh,
            imgsz=imgsz,
            classes=list(COCO_TO_SIH_MAPPING.keys()),
            verbose=False
        )
    else:
        results = model.predict(
            source=str(image_path),
            conf=conf_thresh,
            imgsz=imgsz,
            verbose=False
        )

    res = results[0]

    # Load image for cropping using OpenCV
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image at {image_path}")
    
    img_h, img_w, _ = img_bgr.shape

    vehicle_records = []
    boxes = res.boxes

    print(f"\nTotal Vehicles Detected: {len(boxes)}")
    print("-" * 60)

    for idx, box in enumerate(boxes, start=1):
        raw_cls_id = int(box.cls[0].item())

        if is_coco_model and raw_cls_id in COCO_TO_SIH_MAPPING:
            cls_id, cls_name = COCO_TO_SIH_MAPPING[raw_cls_id]
        else:
            cls_id = raw_cls_id
            cls_name = model.names.get(cls_id, TARGET_CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"))

        confidence = float(box.conf[0].item())

        # Bounding box in integer pixel coordinates [x1, y1, x2, y2]
        x1_f, y1_f, x2_f, y2_f = box.xyxy[0].tolist()
        x1 = max(0, int(round(x1_f)))
        y1 = max(0, int(round(y1_f)))
        x2 = min(img_w, int(round(x2_f)))
        y2 = min(img_h, int(round(y2_f)))

        # Crop vehicle region from image
        crop_bgr = img_bgr[y1:y2, x1:x2]
        crop_filename = f"vehicle_{idx}_{cls_name}.jpg"
        crop_path = crops_dir / crop_filename

        if crop_bgr.size > 0:
            cv2.imwrite(str(crop_path), crop_bgr)
            print(f"  Det #{idx:02d} | Class ID: {cls_id} ({cls_name:<13}) | Conf: {confidence:.4f} | Box: [{x1}, {y1}, {x2}, {y2}] -> Saved Crop: {crop_filename}")
        else:
            print(f"  Det #{idx:02d} | Class ID: {cls_id} ({cls_name:<13}) | Conf: {confidence:.4f} | Warning: Empty crop bbox!")

        record = {
            "vehicle_id": idx,
            "vehicle_class_id": cls_id,
            "vehicle_class": cls_name,
            "vehicle_confidence": round(confidence, 4),
            "bbox": [x1, y1, x2, y2],
            "vehicle_crop_path": str(crop_path.relative_to(output_dir.parent) if output_dir.parent in crop_path.parents else crop_path)
        }
        vehicle_records.append(record)

    # Save annotated visualization image
    annotated_bgr = res.plot()
    annotated_save_path = output_dir / "annotated_detection.jpg"
    cv2.imwrite(str(annotated_save_path), annotated_bgr)
    print(f"\n[OUTPUT] Annotated visualization saved to: {annotated_save_path.resolve()}")

    # Save metadata JSON file
    metadata = {
        "source_image": str(image_path),
        "image_dimensions": {"height": img_h, "width": img_w},
        "total_vehicles_detected": len(vehicle_records),
        "annotated_image_path": str(annotated_save_path.relative_to(output_dir.parent) if output_dir.parent in annotated_save_path.parents else annotated_save_path),
        "vehicles": vehicle_records
    }

    json_save_path = output_dir / "detections.json"
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[OUTPUT] Detection metadata saved to     : {json_save_path.resolve()}")
    print(f"{'='*60}\n")

    return metadata

def main():
    workspace_dir = Path(__file__).parent.resolve()
    input_img = workspace_dir / "input" / "test_image.jpg"

    if not input_img.exists():
        input_dir = workspace_dir / "input"
        image_candidates = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
        if image_candidates:
            input_img = image_candidates[0]
        else:
            print(f"Error: No image found in input directory '{input_dir}'")
            sys.exit(1)

    # Default to yolo11n.pt with 5-class remapping for accurate detection
    model_weights = workspace_dir / "yolo11n.pt"
    if not model_weights.exists():
        custom_weights = workspace_dir / "models" / "vehicle_detector_yolo11n.pt"
        if custom_weights.exists():
            model_weights = custom_weights

    output_dir = workspace_dir / "output"

    results = detect_vehicles(
        image_path=str(input_img),
        model_path=str(model_weights),
        output_dir=str(output_dir),
        conf_thresh=0.25
    )

    print("--- MODEL 1 OUTPUT METADATA JSON ---")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
