"""
==============================================================================
SIH 2026 - MODEL 2: LICENSE PLATE DETECTOR & CROPPING PIPELINE
==============================================================================
This module consumes vehicle detection outputs from Model 1 (crops & JSON metadata),
uses a trained YOLO License Plate Detector to pinpoint license plates within each
vehicle crop, translates crop-relative LP coordinates back into full image space,
extracts individual LP crop images, and outputs structured JSON metadata for 
downstream Model 3 (OCR) and Model 4 (Plate Type/State Classification).
==============================================================================
"""

import os
import sys
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def detect_lp_in_vehicle_crop(
    crop_bgr: np.ndarray,
    model: YOLO,
    conf_thresh: float = 0.20
) -> list:
    """
    Detect license plate bounding box(es) inside a cropped vehicle image using YOLO.

    Returns a list of dicts with crop-relative bounding boxes [x1, y1, x2, y2]
    and confidence scores.
    """
    crop_h, crop_w, _ = crop_bgr.shape
    lp_detections = []

    if model is not None:
        results = model.predict(source=crop_bgr, conf=conf_thresh, verbose=False)
        res = results[0]
        boxes = res.boxes

        for box in boxes:
            confidence = float(box.conf[0].item())
            x1_f, y1_f, x2_f, y2_f = box.xyxy[0].tolist()
            x1 = max(0, int(round(x1_f)))
            y1 = max(0, int(round(y1_f)))
            x2 = min(crop_w, int(round(x2_f)))
            y2 = min(crop_h, int(round(y2_f)))

            if (x2 - x1) > 15 and (y2 - y1) > 8:
                lp_detections.append({
                    "crop_bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 4),
                    "method": "YOLO_LicensePlate"
                })

    return lp_detections

def process_model2(
    model1_metadata_path: str = "output/detections.json",
    lp_model_path: str = "models/license_plate_detector.pt",
    output_dir: str = "output",
    conf_thresh: float = 0.20
) -> dict:
    """
    Main Model 2 pipeline function.
    Reads Model 1 vehicle output metadata, detects license plates on each cropped
    vehicle using the trained LP YOLO model, extracts LP crops, overlays annotations,
    and writes lp_detections.json.
    """
    model1_metadata_path = Path(model1_metadata_path)
    output_dir = Path(output_dir)

    if not model1_metadata_path.exists():
        raise FileNotFoundError(f"Model 1 metadata not found at: {model1_metadata_path.resolve()}")

    with open(model1_metadata_path, "r", encoding="utf-8") as f:
        m1_data = json.load(f)

    source_image_path = Path(m1_data.get("source_image", "input/test_image.jpg"))
    if not source_image_path.exists():
        source_image_path = Path("input/test_image.jpg")

    full_img_bgr = cv2.imread(str(source_image_path))
    if full_img_bgr is None:
        raise ValueError(f"Could not load full source image: {source_image_path}")

    full_h, full_w, _ = full_img_bgr.shape
    annotated_full_bgr = full_img_bgr.copy()

    # Load YOLO LP detector model
    lp_model_path = Path(lp_model_path)
    if not lp_model_path.exists():
        fallback_path = Path("models/license_plate_detector_yolov8.pt")
        if fallback_path.exists():
            lp_model_path = fallback_path

    print(f"\n{'='*60}")
    print("MODEL 2: LICENSE PLATE DETECTION & CROPPING PIPELINE")
    print(f"{'='*60}")
    print(f"Loading LP Model Weights: {lp_model_path.resolve()}")
    print(f"Model 1 Metadata File  : {model1_metadata_path.resolve()}")
    print(f"Total Vehicles Input   : {len(m1_data.get('vehicles', []))}")

    lp_model = YOLO(str(lp_model_path))

    lp_crops_dir = output_dir / "license_plates"
    if lp_crops_dir.exists():
        shutil.rmtree(lp_crops_dir)
    lp_crops_dir.mkdir(parents=True, exist_ok=True)

    vehicles_output = []
    total_lps_detected = 0

    for vehicle in m1_data.get("vehicles", []):
        v_id = vehicle["vehicle_id"]
        v_class = vehicle["vehicle_class"]
        v_conf = vehicle["vehicle_confidence"]
        v_bbox = vehicle["bbox"]  # [x1, y1, x2, y2]
        vx1, vy1, vx2, vy2 = v_bbox

        crop_rel_path = vehicle.get("vehicle_crop_path")
        crop_abs_path = output_dir.parent / crop_rel_path if not os.path.isabs(crop_rel_path) else Path(crop_rel_path)

        if not crop_abs_path.exists():
            print(f"[WARNING] Vehicle crop file missing: {crop_abs_path}")
            continue

        vehicle_crop_bgr = cv2.imread(str(crop_abs_path))
        if vehicle_crop_bgr is None:
            continue

        # Draw vehicle bounding box on full annotated image (Green)
        cv2.rectangle(annotated_full_bgr, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
        cv2.putText(annotated_full_bgr, f"Vehicle #{v_id}: {v_class}", (vx1, max(15, vy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect License Plates inside vehicle crop
        lp_results = detect_lp_in_vehicle_crop(vehicle_crop_bgr, model=lp_model, conf_thresh=conf_thresh)

        lp_records = []
        for lp_idx, lp_item in enumerate(lp_results, start=1):
            cx1, cy1, cx2, cy2 = lp_item["crop_bbox"]
            lp_conf = lp_item["confidence"]
            detection_method = lp_item["method"]

            # Translate crop coordinates back to original full image space
            fx1 = vx1 + cx1
            fy1 = vy1 + cy1
            fx2 = vx1 + cx2
            fy2 = vy1 + cy2

            # Crop License Plate image from vehicle crop
            lp_crop_bgr = vehicle_crop_bgr[cy1:cy2, cx1:cx2]
            lp_filename = f"lp_vehicle_{v_id}_{v_class}_{lp_idx}.jpg"
            lp_save_path = lp_crops_dir / lp_filename

            if lp_crop_bgr.size > 0:
                cv2.imwrite(str(lp_save_path), lp_crop_bgr)

            # Draw License Plate bounding box on full annotated image (Yellow)
            cv2.rectangle(annotated_full_bgr, (fx1, fy1), (fx2, fy2), (0, 255, 255), 3)
            cv2.putText(annotated_full_bgr, f"LP #{total_lps_detected + 1} ({lp_conf:.2f})",
                        (fx1, max(15, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            total_lps_detected += 1

            lp_records.append({
                "lp_id": lp_idx,
                "lp_confidence": lp_conf,
                "detection_method": detection_method,
                "crop_relative_bbox": [cx1, cy1, cx2, cy2],
                "full_image_bbox": [fx1, fy1, fx2, fy2],
                "lp_crop_path": str(lp_save_path.relative_to(output_dir.parent) if output_dir.parent in lp_save_path.parents else lp_save_path)
            })

            print(f"  Vehicle #{v_id:02d} ({v_class:<10}) -> Detected License Plate | Conf: {lp_conf:.4f} | BBox: [{fx1}, {fy1}, {fx2}, {fy2}] -> Saved Crop: {lp_filename}")

        vehicle_entry = dict(vehicle)
        vehicle_entry["license_plates"] = lp_records
        vehicles_output.append(vehicle_entry)

    # Save full annotated image with vehicle & LP bounding boxes
    annotated_save_path = output_dir / "annotated_license_plates.jpg"
    cv2.imwrite(str(annotated_save_path), annotated_full_bgr)
    print(f"\n[OUTPUT] Annotated full image saved to    : {annotated_save_path.resolve()}")

    # Output structured JSON metadata for Model 3 & Model 4
    output_metadata = {
        "source_image": str(source_image_path),
        "image_dimensions": {"height": full_h, "width": full_w},
        "total_vehicles_processed": len(vehicles_output),
        "total_license_plates_detected": total_lps_detected,
        "annotated_image_path": str(annotated_save_path.relative_to(output_dir.parent) if output_dir.parent in annotated_save_path.parents else annotated_save_path),
        "vehicles": vehicles_output
    }

    json_save_path = output_dir / "lp_detections.json"
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=4)

    print(f"[OUTPUT] License Plate metadata saved to : {json_save_path.resolve()}")
    print(f"{'='*60}\n")

    return output_metadata

def main():
    workspace_dir = Path(__file__).parent.resolve()
    m1_json = workspace_dir / "output" / "detections.json"

    if not m1_json.exists():
        print(f"Error: Model 1 output metadata file not found at '{m1_json}'.")
        print("Please run model1.py first to generate vehicle crops and detections.json.")
        sys.exit(1)

    lp_weights = workspace_dir / "models" / "license_plate_detector.pt"
    out_dir = workspace_dir / "output"

    results = process_model2(
        model1_metadata_path=str(m1_json),
        lp_model_path=str(lp_weights),
        output_dir=str(out_dir),
        conf_thresh=0.20
    )

    print("--- MODEL 2 OUTPUT METADATA JSON ---")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
