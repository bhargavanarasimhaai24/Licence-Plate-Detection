# SIH 2026 - Vehicle & License Plate Detection Pipeline

An end-to-end Computer Vision pipeline built for **Smart India Hackathon (SIH) 2026**. This system detects vehicles in full-resolution images, crops detected vehicles, pinpoints license plates using a secondary specialized detector, and exports structured metadata JSON alongside annotated visualization images.

---

## 🚀 Pipeline Architecture

```
Input Image ──> [ Model 1: Vehicle Detector ] ──> Vehicle Crops & detections.json
                               │
                               ▼
                    [ Model 2: LP Detector ] ──> LP Crops & lp_detections.json
```

1. **Model 1 (`model1.py`)**: Performs vehicle detection on input images using YOLO11n. Extracts individual vehicle crops (`auto_rickshaw`, `car`, `bus`, `motorcycle`, `truck`) and generates bounding box metadata.
2. **Model 2 (`model2.py`)**: Consumes cropped vehicle images from Model 1, detects license plates with a custom-trained YOLO LP detector, translates plate coordinates back to original full image coordinates, and extracts cropped license plate images.

---

## 📈 Model Performance & Accuracy

| Model Stage | Architecture | mAP@50 | mAP@50-95 | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model 1 (Vehicle Detection)** | YOLO11n (`models/vehicle_detector_yolo11n.pt`) | **81.09%** | **56.53%** | **71.14%** | **80.35%** |
| **Model 2 (License Plate Detection)** | Custom YOLO (`models/license_plate_detector.pt`) | **93.60%** | **74.52%** | **94.97%** | **90.00%** |

---

## 📁 Repository Structure

```
SIH_2026/
├── input/                          # Input images directory (place input images here)
│   └── test_image.jpg
├── models/                         # Pre-trained model weights
│   ├── vehicle_detector_yolo11n.pt # Custom Model 1 YOLO11n weights
│   └── license_plate_detector.pt   # Model 2 LP YOLO detector weights
├── output/                         # Generated output directory
│   ├── crops/                      # Cropped vehicle images from Model 1
│   ├── license_plates/             # Cropped license plate images from Model 2
│   ├── annotated_detection.jpg     # Model 1 visual overlay
│   ├── annotated_license_plates.jpg# Model 2 visual overlay
│   ├── detections.json             # Vehicle detection metadata
│   └── lp_detections.json          # License plate detection metadata
├── model1.py                       # Model 1 Vehicle Detection script
├── model2.py                       # Model 2 License Plate Detection script
├── download_lp_dataset.py          # Roboflow dataset downloader script
├── yolo11n.pt                      # Fallback COCO YOLO11 base weights
└── README.md                       # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install ultralytics opencv-python numpy roboflow torch
```

---

## 🏃 How to Run

### Step 1: Prepare Input Image
Place your input image(s) into the `input/` folder (e.g., `input/test_image.jpg`).

### Step 2: Run Model 1 (Vehicle Detection)
Run Model 1 to detect vehicles, crop them, and generate `output/detections.json`:

```bash
python model1.py
```

### Step 3: Run Model 2 (License Plate Detection)
Run Model 2 to detect license plates inside the vehicle crops and generate `output/lp_detections.json`:

```bash
python model2.py
```

---

## 📊 Dataset Download (Optional)

To download the Roboflow License Plate dataset for model re-training or evaluation:

```bash
python download_lp_dataset.py
```

---

## 📄 Output Data Format

* **`output/detections.json`**: Contains bounding box coordinates, vehicle class IDs, confidence scores, and vehicle crop paths.
* **`output/lp_detections.json`**: Contains vehicle records along with crop-relative and full-image license plate bounding boxes, confidence scores, and LP crop paths.
