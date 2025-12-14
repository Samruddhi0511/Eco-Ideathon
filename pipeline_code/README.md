# Rooftop Solar PV Verification Pipeline

## Overview

This repository implements an end-to-end **Rooftop Solar Photovoltaic (PV) Verification Pipeline** in Python.

The pipeline ingests rooftop coordinates, retrieves high-resolution satellite imagery, applies a YOLO-based segmentation model to detect rooftop solar panels, validates detections using spatial buffer logic, and produces **auditable and explainable outputs**.

The system is designed to be **deterministic, transparent, and scalable**, suitable for large-scale rooftop verification under diverse imaging conditions.

---

## Key Features

- Ingests rooftop coordinates from CSV or XLSX files  
- Retrieves satellite imagery using **Google Static Maps**  
- Applies spatial jittering to handle geolocation noise  
- Performs image quality control (cloud glare, shadows, resolution)  
- Runs YOLO-based **segmentation inference**  
- Validates detections using rooftop buffer logic:
  - 1200 sqft (primary)
  - 2400 sqft (fallback)
- Estimates:
  - PV area (m²)
  - Panel count
  - Installed capacity (kW)
- Outputs **GeoJSON-compatible polygon masks**
- Saves audit-ready visual overlays and artifacts  

---

## Input Data Format

### Supported Formats
- `.xlsx`
- `.csv`

### Required Columns

| Column Name | Description |
|------------|------------|
| `sample_id` | Unique identifier for the rooftop |
| `latitude` | Latitude (WGS84) |
| `longitude` | Longitude (WGS84) |

---

## Pipeline Workflow

### 1. Image Acquisition

- Retrieves satellite imagery centered at the provided coordinates  
- Applies ±10 m spatial jitter to mitigate geocoding inaccuracies  
- Selects the best image using a heuristic quality score  

---

### 2. Image Quality Control (QC)

Each image is evaluated for:

- Low spatial resolution  
- Excessive brightness (cloud glare)  
- Excessive darkness (shadowing or occlusion)  

Images failing QC thresholds are marked as **NOT_VERIFIABLE**, ensuring conservative and auditable decision-making.

---

### 3. Segmentation Inference

- Uses Ultralytics YOLO **segmentation models**  
- Extracts pixel-level segmentation masks for detected PV arrays  
- Falls back to bounding boxes if segmentation masks are unavailable  
- Filters detections using minimum area and confidence thresholds  

---

### 4. Rooftop Buffer Validation

Detected PV regions are validated using concentric circular buffers centered on the input coordinate:

- **1200 sqft buffer** is evaluated first  
- **2400 sqft buffer** is evaluated if no PV is found in the smaller buffer  

If multiple PV regions overlap a buffer, the region with the **largest overlap area** is selected.

---

### 5. Solar Estimation

For validated detections, the pipeline computes:

- Estimated PV panel area (m²)  
- Estimated number of panels  
- Estimated installed capacity (kW), using documented assumptions  
- Area-weighted, calibrated confidence score  

All estimates are derived transparently from the detected geometry and recorded assumptions.

---

## Outputs

### Prediction File

The pipeline produces a consolidated JSON file containing one record per rooftop, including:

- Detection result (`has_solar`)
- Confidence score
- PV area, panel count, and capacity estimates
- Buffer radius used
- QC status (`VERIFIABLE` / `NOT_VERIFIABLE`)
- Polygon mask or bounding geometry
- Image metadata and assumptions

### Audit Artifacts

For each rooftop, the following artifacts are saved:

- Raw satellite image  
- QC-annotated image  
- Buffer visualization images  
- Segmentation mask image  
- Final audit overlay with geometry and metrics  

---

---

## Running the Pipeline
## API Key Configuration (Required)

Satellite imagery is fetched using **Google Static Maps**, which requires a valid API key with billing enabled.

**API keys are not included in this repository.**

### Step 1: Create a Google Static Maps API Key

1. Go to Google Cloud Console  
2. Create or select a project  
3. Enable **Maps Static API**  
4. Enable billing  
5. Generate an API key  
6. (Recommended) Restrict the key:
   - Limit to Maps Static API
   - Apply a billing or usage cap

The pipeline can be executed with or without explicitly passing the Google Maps API key, depending on how the key is provided.

---

### Step 2: Provide the API Key (Two Supported Methods)

#### Option A: Environment Variable (Recommended)

This is the preferred and secure method, especially for server execution.

**Linux / Server**
```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_MAPS_API_KEY"
```
### Case 1: API Key Passed via Command-Line Argument

Use this method if the API key is **not set as an environment variable**.

```bash
python infer.py \
  --input_file data/input.csv \
  --output_dir outputs/run_01 \
  --model_path models/yolo_seg.pt \
  --api_key YOUR_GOOGLE_MAPS_API_KEY
```
Use this method if the API key is already set in the environment as GOOGLE_API_KEY

```bash
python infer.py \
  --input_file data/input.csv \
  --output_dir outputs/run_01 \
  --model_path models/yolo_seg.pt
```

### Arguments
### Command-Line Arguments

| Argument | Description |
|--------|------------|
| `--input_file` | Path to the input XLSX/CSV file containing `sample_id`, `latitude`, and `longitude` |
| `--output_dir` | Directory where prediction JSON and audit artifacts will be saved |
| `--model_path` | Path to the provided YOLO segmentation model (`.pt`) included in this repository |
| `--api_key` | Google Static Maps API key (optional; can also be provided via environment variable) |

**Note:**  
A trained YOLO segmentation model (`.pt`) is included as part of the submission.  
No model training is required to run inference.

## Failure Handling

The pipeline robustly handles:<br>

-Poor image quality<br>

-Inference errors<br>

Such cases are explicitly marked as NOT_VERIFIABLE with reason codes.<br>

## Design Principles

-Conservative decision-making under uncertainty<br>

-Deterministic and reproducible inference<br>

-Explicit confidence calibration<br>

-Geometry-aware validation<br>

-Audit-friendly outputs<br>
## Outputs

###  Prediction File

The pipeline generates a consolidated prediction file:

Each record includes:

```json
{
  "sample_id": "ID_001",
  "lat": 12.9716,
  "lon": 77.5946,
  "has_solar": true,
  "confidence": 0.82,
  "pv_area_sqm_est": 18.4,
  "panel_count_est": 10,
  "capacity_kw_est": 3.5,
  "buffer_radius_sqft": 1200,
  "qc_status": "VERIFIABLE",
  "bbox_or_mask": {
    "type": "Polygon",
    "coordinates": [...]
  },
  "image_metadata": {
    "source": "Google Static Maps",
    "retrieved_at": "2025-01-01T12:00:00Z"
  }
}

```
## License
This project is intended for academic, research, and evaluation use.




