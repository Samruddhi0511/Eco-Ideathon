#!/usr/bin/env python3
import os
import math
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageStat
from io import BytesIO
from shapely.geometry import Polygon, Point, mapping, shape
from shapely.ops import unary_union
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# Encapsulated constants for easier management and clarity.
# =========================================================
class PipelineConfig:
    IMG_SIZE_PX = 512          # Desired image dimension (square)
    DEFAULT_ZOOM = 20           # Default zoom level for imagery
    IMG_SCALE = 2               # Google Static Maps scale (1 or 2 for retina)
    JITTER_METERS = 10          # Radius for coordinate jittering
    REQUEST_TIMEOUT_SECONDS = (5, 10) # (connect_timeout, read_timeout) for HTTP requests

    CONF_THRESHOLD = 0.25       # Minimum confidence to consider a detection
    MIN_POLY_AREA_PX = 30       # Minimum pixel area for a valid polygon

    # QC thresholds for image quality assessment
    BRIGHTNESS_CLOUDY_THRESHOLD = 220 # Pixels above this are considered bright/cloudy
    DARK_THRESHOLD = 25         # Pixels below this are considered dark/shadowed

    # Capacity + panel assumptions (documented in metadata for auditability)
    WP_PER_M2 = 190             # Watt peak per square meter assumption (crystalline silicon)
    PANEL_AREA_M2 = 1.9         # Typical rooftop panel area in m^2 (approx 1.7-2.0)

    # Calibration helper
    CALIB_AREA_SATURATE_M2 = 5.0 # PV area (m^2) where confidence calibration saturates

    # Buffer search radii in square feet (converted to meters and then pixels)
    BUFFER_SEARCH_SQFT = [1200, 2400]

    # Placeholder for actual image capture date if not available from source metadata.
    # IMPORTANT: The current image fetching methods (Google Static Maps, ESRI)
    # generally do not provide explicit capture dates. The pipeline uses the
    # *processing date* as a placeholder. For true auditability, a different
    # imagery source (e.g., Google Earth Engine, commercial APIs) that provides
    # this metadata would be required.
    PROCESSING_DATE_PLACEHOLDER = time.strftime("%Y-%m-%d")
 # Indicate it's processing date
    # Or, if we strictly adhere to 'capture_date', then we must acknowledge it's an unknown and use None

# =========================================================
# LOGGING UTILITY
# =========================================================
def log(stage: str, msg: str):
    """Prints a formatted log message."""
    print(f"[{stage.upper()}] {msg}")

# =========================================================
# GEOSPATIAL AND IMAGE UTILITIES
# =========================================================
def meters_per_pixel(
    lat: float,
    zoom: int = PipelineConfig.DEFAULT_ZOOM
) -> float:
    """
    Calculates ground resolution (meters per pixel) at a given latitude
    for Web Mercator imagery, accounting for retina scale.
    """
    R = 6378137.0  # Earth's radius in meters

    tile_px = 256 * PipelineConfig.IMG_SCALE
    return (
        math.cos(math.radians(lat)) * 2 * math.pi * R
    ) / (tile_px * (2 ** zoom))


def meters_to_latlon_offset(lat: float, meters: float) -> tuple[float, float]:
    """
    Converts a distance in meters to approximate latitude and longitude offsets.
    """
    # Approximate conversion factors for small distances
    dlat = meters / 110574.0  # ~meters per degree latitude
    dlons_at_equator = 111320.0 # ~meters per degree longitude at equator
    if math.cos(math.radians(lat)) == 0: # Avoid division by zero at poles
        dlon = 0.0
    else:
        dlon = meters / (dlons_at_equator * math.cos(math.radians(lat)))
    return dlat, dlon

def generate_jitters(lat: float, lon: float, jitter_meters: int = PipelineConfig.JITTER_METERS) -> list[tuple[float, float]]:
    """
    Generates a list of (lat, lon) coordinates, including the original and
    offsets within a specified jitter radius.
    """
    dlat, dlon = meters_to_latlon_offset(lat, jitter_meters)
    return [
        (lat, lon),
        (lat + dlat, lon), (lat - dlat, lon),
        (lat, lon + dlon), (lat, lon - dlon)
    ]

# =========================================================
# IMAGE QUALITY CONTROL (QC)
# =========================================================
def basic_qc(img: Image.Image) -> dict:
    """
    Performs basic quality control on an image (resolution, brightness).
    Returns a dictionary with brightness mean and detected flags.
    """
    width, height = img.size
    mean_brightness = ImageStat.Stat(img.convert("L")).mean[0] # Average brightness of grayscale image
    flags = []

    if width < PipelineConfig.IMG_SIZE_PX or height < PipelineConfig.IMG_SIZE_PX:
        flags.append("LOW_RESOLUTION")
    if mean_brightness > PipelineConfig.BRIGHTNESS_CLOUDY_THRESHOLD:
        flags.append("CLOUD_GLARE")
    if mean_brightness < PipelineConfig.DARK_THRESHOLD:
        flags.append("DARK_SHADOW")
    return {"brightness": mean_brightness, "flags": flags}

def qc_score(qc_result: dict) -> float:
    """
    Calculates a heuristic QC score based on flags and brightness.
    Lower score is better.
    """
    score = len(qc_result["flags"]) * 5.0 # Penalize each flag
    # Penalize deviation from mid-brightness (e.g., 120-130 is neutral grey)
    score += abs(qc_result["brightness"] - 120.0) / 40.0
    return score

def save_qc_image(img_np: np.ndarray, qc_result: dict, out_path: str):
    """
    Saves an image with QC flags overlaid as text.
    """
    vis_img = img_np.copy()
    y_offset = 30
    for flag in qc_result["flags"]:
        cv2.putText(vis_img, flag, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 30
    cv2.imwrite(out_path, vis_img)

# =========================================================
# IMAGE FETCHING
# Prioritizes Google Static Maps, falls back to ESRI World Imagery.
# =========================================================
def fetch_google_static_maps(lat: float, lon: float, api_key: str) -> tuple[Image.Image, str]:
    """
    Fetches a ~1024x1024 satellite image using Google Static Maps
    via scale=2 (retina) safely.
    """
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY_NOT_SET")

    # Google limit: max 640x640 per scale unit
    # scale=2 → 512x512 → 1024x1024 actual pixels
    logical_size = PipelineConfig.IMG_SIZE_PX // PipelineConfig.IMG_SCALE

    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": PipelineConfig.DEFAULT_ZOOM,
        "size": f"{logical_size}x{logical_size}",
        "scale": PipelineConfig.IMG_SCALE,
        "maptype": "satellite",
        "key": api_key
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (SolarVerificationPipeline)"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=PipelineConfig.REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            raise RuntimeError(f"GOOGLE_NON_IMAGE_RESPONSE: {content_type}")

        img = Image.open(BytesIO(response.content)).convert("RGB")

        # 🔒 Hard safety check
        if img.size != (PipelineConfig.IMG_SIZE_PX, PipelineConfig.IMG_SIZE_PX):
            raise RuntimeError(
                f"GOOGLE_IMAGE_SIZE_MISMATCH: got {img.size}, "
                f"expected {(PipelineConfig.IMG_SIZE_PX, PipelineConfig.IMG_SIZE_PX)}"
            )

        return img, "Google Static Maps (scale=2)"

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            raise RuntimeError("GOOGLE_403_FORBIDDEN: API key / billing / API enablement issue") from e
        if e.response.status_code == 429:
            raise RuntimeError("GOOGLE_RATE_LIMIT") from e
        raise RuntimeError(f"GOOGLE_HTTP_ERROR_{e.response.status_code}") from e

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"GOOGLE_REQUEST_FAILED: {e}") from e


def fetch_esri_world_imagery(lat: float, lon: float) -> tuple[Image.Image, str]:
    """
    Fetches an image from ESRI World Imagery service.
    Raises RuntimeError on failure.
    """
    try:
        mpp = meters_per_pixel(lat)
        # Calculate bounding box for the desired image size
        half_m = (PipelineConfig.IMG_SIZE_PX / 2.0) * mpp
        dlat_bbox, dlon_bbox = meters_to_latlon_offset(lat, half_m)

        # BBOX format: min_x, min_y, max_x, max_y (lon, lat)
        bbox = f"{lon - dlon_bbox},{lat - dlat_bbox},{lon + dlon_bbox},{lat + dlat_bbox}"

        url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        params = {
            "bbox": bbox,
            "bboxSR": "4326",  # WGS84
            "imageSR": "3857", # Web Mercator
            "size": f"{PipelineConfig.IMG_SIZE_PX},{PipelineConfig.IMG_SIZE_PX}",
            "format": "png",
            "f": "image"
        }
        headers = {"User-Agent": "Mozilla/5.0 (SolarVerificationPipeline)"}

        response = requests.get(url, params=params, headers=headers, timeout=PipelineConfig.REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img, "ESRI World Imagery"
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"ESRI_FETCH_FAILED: {e}") from e

def get_best_image(original_lat: float, original_lon: float, api_key: str | None) -> dict | None:
    """
    Attempts to fetch an image from Google and then ESRI, trying jittered coordinates.
    Returns the best quality image (lowest QC score) or None if all attempts fail.
    """
    candidates = []
    jitter_attempts = generate_jitters(original_lat, original_lon, PipelineConfig.JITTER_METERS)

    for i, (jlat, jlon) in enumerate(jitter_attempts):
        log("FETCH", f"Attempt {i+1}/{len(jitter_attempts)} for ({original_lat:.4f},{original_lon:.4f}) at jitter ({jlat:.4f},{jlon:.4f})")

        # Try Google Static Maps first if API key is provided
        if api_key:
            try:
                img, src = fetch_google_static_maps(jlat, jlon, api_key)
                qc = basic_qc(img)
                candidates.append({
                    "img": img, "lat": jlat, "lon": jlon, "qc": qc,
                    "score": qc_score(qc), "source": src
                })
                log("FETCH", f"Google OK (jitter {i+1})")
                # If the first attempt (original lat/lon) is successful and high quality, might not need more.
                # However, for robustness, we complete all jitters and pick the best.
            except RuntimeError as e:
                log("FETCH_GOOGLE_FAIL", f"({jlat},{jlon}) -> {e}")
            except Exception as e: # Catch other unexpected errors
                log("FETCH_GOOGLE_FAIL", f"({jlat},{jlon}) -> Unexpected error: {e}")

        # Always try ESRI as a fallback
        try:
            img, src = fetch_esri_world_imagery(jlat, jlon)
            qc = basic_qc(img)
            candidates.append({
                "img": img, "lat": jlat, "lon": jlon, "qc": qc,
                "score": qc_score(qc), "source": src
            })
            log("FETCH", f"ESRI OK (jitter {i+1})")
        except RuntimeError as e:
            log("FETCH_ESRI_FAIL", f"({jlat},{jlon}) -> {e}")
        except Exception as e: # Catch other unexpected errors
            log("FETCH_ESRI_FAIL", f"({jlat},{jlon}) -> Unexpected error: {e}")

    if not candidates:
        log("FETCH", "All jitter attempts failed for all sources.")
        return None

    best = min(candidates, key=lambda x: x["score"])
    log("FETCH", f"Selected image from {best['source']} with QC score {best['score']:.2f}")
    return best

# =========================================================
# MASK TO POLYGON CONVERSION & ENCODING
# =========================================================
def mask_to_polygon(mask_uint8: np.ndarray) -> Polygon | None:
    """
    Converts a binary mask (uint8, 0/1) to a shapely Polygon.
    Combines multiple disconnected components into a single MultiPolygon or Polygon.
    Filters out very small polygons.
    """
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3: # A polygon needs at least 3 points
            try:
                # Squeeze to remove redundant dimensions (e.g., (N, 1, 2) -> (N, 2))
                poly = Polygon(contour.squeeze())
                if poly.is_valid and poly.area >= PipelineConfig.MIN_POLY_AREA_PX:
                    polygons.append(poly)
            except Exception as e:
                log("MASK_TO_POLY_WARN", f"Error creating polygon from contour: {e}")
                continue
    if not polygons:
        return None
    # Combine all valid polygons into a single (potentially Multi)Polygon
    return unary_union(polygons)

def ensure_single_polygon(poly: Polygon | None) -> Polygon | None:
    """
    If a MultiPolygon, returns the largest component. Otherwise, returns the Polygon itself.
    Handles None input gracefully.
    """
    if poly is None:
        return None
    if poly.geom_type == "Polygon":
        return poly
    if poly.geom_type == "MultiPolygon":
        try:
            # Return the component polygon with the largest area
            return max(poly.geoms, key=lambda g: g.area)
        except Exception as e:
            log("POLY_PROCESS_WARN", f"Error processing MultiPolygon: {e}")
            return None
    log("POLY_PROCESS_WARN", f"Unexpected geometry type: {poly.geom_type}")
    return None

def encode_polygon_to_geojson_string(poly: Polygon | None) -> str | None:
    """
    Encodes a shapely Polygon (or MultiPolygon) to a GeoJSON string.
    Returns None if the input polygon is None.
    """
    if poly is None:
        return None
    try:
        return json.dumps(mapping(poly))
    except Exception as e:
        log("GEOJSON_ENCODE_ERROR", f"Failed to encode polygon to GeoJSON: {e}")
        return None

def save_mask_image(img_np: np.ndarray, poly: Polygon | None, out_path: str):
    """
    Saves an image representing the detected polygon mask.
    """
    overlay = np.zeros_like(img_np)
    poly_single = ensure_single_polygon(poly)
    if poly_single is not None:
        # Exterior points are (x,y)
        # OpenCV expects integer points, so convert and reshape if necessary
        exterior_pts = np.array(poly_single.exterior.coords).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [exterior_pts], (0, 255, 0)) # Green fill
    cv2.imwrite(out_path, overlay)

def save_buffer_image(img_np: np.ndarray, center_pt: Point, radius_px: float, out_path: str):
    """
    Saves an image with a circle drawn at the center to represent the buffer search area.
    """
    vis_img = img_np.copy()
    # OpenCV circle center is (x,y) tuple, radius is int
    cv2.circle(vis_img, (int(center_pt.x), int(center_pt.y)), int(radius_px), (0, 0, 255), 2) # Red circle
    cv2.imwrite(out_path, vis_img)

# =========================================================
# EXPLANATION & CALIBRATION HELPERS
# =========================================================
def calibrate_confidence(raw_conf: float, area_m2: float) -> float:
    """
    Applies a transparent area-weighted confidence calibration.
    NOTE: This is a heuristic calibration, not a rigorous statistical one
    like Platt scaling or isotonic regression.
    """
    if area_m2 <= 0:
        return 0.0
    area_factor = min(1.0, area_m2 / PipelineConfig.CALIB_AREA_SATURATE_M2)
    calibrated = raw_conf * (0.6 + 0.4 * area_factor) # Increase confidence for larger areas
    calibrated = max(0.0, min(0.995, calibrated)) # Clamp between 0 and 0.995
    return round(calibrated, 3)

def estimate_panel_count(area_m2: float) -> int:
    """
    Estimates the number of solar panels based on total PV area.
    """
    if area_m2 <= 0:
        return 0
    return max(1, int(round(area_m2 / PipelineConfig.PANEL_AREA_M2)))

def estimate_capacity_kw(area_m2: float) -> float:
    """
    Estimates the capacity in kW based on total PV area and Wp/m^2 assumption.
    """
    if area_m2 <= 0:
        return 0.0
    return round((area_m2 * PipelineConfig.WP_PER_M2) / 1000.0, 3)

def reason_codes_for(poly: Polygon | None, confidence: float, qc_flags: list[str]) -> list[str]:
    """
    Generates a list of reason codes for explainability.
    """
    reasons = []

    # Imaging quality reasons
    if "CLOUD_GLARE" in qc_flags or "DARK_SHADOW" in qc_flags:
        reasons.append("imaging_artifact")
    if "LOW_RESOLUTION" in qc_flags:
        reasons.append("low_resolution_imagery")

    # Geometric properties of the detected PV array
    poly_single = ensure_single_polygon(poly)
    if poly_single is not None:
        try:
            # Rectangularity check: how close is the polygon to a perfect rectangle
            min_rect = poly_single.minimum_rotated_rectangle
            if min_rect and min_rect.area > 0: # Ensure minimum_rotated_rectangle is valid and has area
                rectangularity = poly_single.area / min_rect.area
                if rectangularity > 0.85:
                    reasons.append("rectilinear_array")
                elif rectangularity > 0.6:
                    reasons.append("complex_array_shape") # Suggests multiple panels or irregular arrangement
            else:
                reasons.append("irregular_shape") # Could indicate noise or complex structure

            # Size-based reasoning
            if poly_single.area > (PipelineConfig.MIN_POLY_AREA_PX * 5): # Arbitrary threshold for "large enough"
                reasons.append("area_consistent_with_pv")
            else:
                reasons.append("small_detected_area")

        except Exception as e:
            log("REASON_CODE_WARN", f"Error calculating polygon reasons: {e}")
            reasons.append("polygon_analysis_failed")

    # Model confidence level
    if confidence >= 0.9:
        reasons.append("high_model_confidence")
    elif confidence >= 0.7:
        reasons.append("moderate_model_confidence")
    else:
        reasons.append("low_model_confidence")

    if not reasons: # Fallback if no specific reasons are generated
        reasons.append("visual_pattern_match")

    # Remove duplicates and limit the number of reason codes for conciseness
    return list(dict.fromkeys(reasons))[:4] # Use dict to maintain order and remove duplicates

# =========================================================
# MAIN PIPELINE CLASS
# Encapsulates the core logic for processing each sample.
# =========================================================
class SolarVerificationPipeline:
    def __init__(self, model_path: str, output_base_dir: str, api_key: str | None = None):
        """
        Initializes the solar verification pipeline.

        Args:
            model_path: Path to the YOLO segmentation model file.
            output_base_dir: Base directory for all output files (JSON, artifacts).
            api_key: Google Static Maps API key (optional).
        """
        self.model = YOLO(model_path)
        self.output_base_dir = output_base_dir
        self.api_key = api_key
        self.artifacts_dir = os.path.join(output_base_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        log("INIT", f"Pipeline initialized. Model: {model_path}, Output: {output_base_dir}")

    def _create_empty_record(self, sample_id: int, lat: float, lon: float, src: str, qc_result: dict, reason_override: list[str] | None = None) -> dict:
        """
        Helper to create a default record when no solar panels are found or
        image acquisition fails.
        """
        current_processing_date = time.strftime("%Y-%m-%d")
        qc_status = "VERIFIABLE" if not qc_result["flags"] else "NOT_VERIFIABLE"
        has_solar_status = False if qc_status == "VERIFIABLE" and not reason_override else None # If verifiable and no other issues, then definitely no solar. Else, unknown.

        default_reasons = ["no_pv_detected"]
        if qc_status == "NOT_VERIFIABLE":
            default_reasons.append("insufficient_image_quality")

        return {
            "sample_id": sample_id,
            "lat": lat,
            "lon": lon,
            "has_solar": has_solar_status,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "panel_count_est": 0,
            "capacity_kw_est": 0.0,
            "buffer_radius_sqft": PipelineConfig.BUFFER_SEARCH_SQFT[-1], # Max buffer tried
            "qc_status": qc_status,
            "reason_codes": reason_override if reason_override else default_reasons,
            "bbox_or_mask": None,
            "image_metadata": {
                "source": src,
                "capture_date": PipelineConfig.PROCESSING_DATE_PLACEHOLDER, # Placeholder
                "capacity_assumption_wp_per_m2": PipelineConfig.WP_PER_M2,
                "confidence_calibrated": False,
                "confidence_method": "none"
            }
        }

    def process(self, row: pd.Series) -> dict:
        """
        Processes a single geographic coordinate (latitude, longitude) to detect solar panels.

        Args:
            row: A pandas Series containing 'sample_id', 'latitude', and 'longitude'.

        Returns:
            A dictionary conforming to the specified output schema.
        """
        sample_id = int(row["sample_id"])
        original_lat = float(row["latitude"])
        original_lon = float(row["longitude"])

        log("PROCESS", f"Starting sample_id={sample_id} at ({original_lat:.4f},{original_lon:.4f})")

        # 1. Fetch Image
        best_image_data = get_best_image(original_lat, original_lon, self.api_key)
        if best_image_data is None:
            log("PROCESS", f"Image acquisition failed for sample {sample_id}")
            return self._create_empty_record(
                sample_id, original_lat, original_lon, "N/A",
                {"brightness": 0, "flags": ["NO_IMAGE_AVAILABLE"]},
                ["image_acquisition_failed"]
            )

        img_pil = best_image_data["img"]
        img_np = np.array(img_pil) # Convert PIL image to NumPy array for OpenCV operations
        current_qc = best_image_data["qc"]
        image_source = best_image_data["source"]
        effective_lat = best_image_data["lat"] # Actual lat from jittering
        effective_lon = best_image_data["lon"] # Actual lon from jittering
        current_processing_date = time.strftime("%Y-%m-%d") # The date the script is run

        # Save raw and QC-flagged images for auditing
        raw_image_path = os.path.join(self.artifacts_dir, f"{sample_id}_raw.png")
        img_pil.save(raw_image_path)
        save_qc_image(img_np, current_qc, os.path.join(self.artifacts_dir, f"{sample_id}_qc.png"))
        log("ARTIFACTS", f"Saved raw and QC images for {sample_id}")

        # Determine image dimensions for model output processing
        img_height, img_width = img_np.shape[:2]

        # 2. Classify (Run Inference)
        log("INFER", f"Running model on sample {sample_id}...")
        # Ultralytics YOLO accepts image path directly, or a PIL/Numpy image
        results = self.model(raw_image_path, imgsz=PipelineConfig.IMG_SIZE_PX,
                             conf=PipelineConfig.CONF_THRESHOLD, verbose=False)

        detected_panels = []
        if results and len(results) > 0:
            result = results[0] # Usually only one result object for a single image input

            # Prefer segmentation masks if available
            if hasattr(result, "masks") and result.masks is not None:
                try:
                    # masks.data is a tensor of masks, shape (N, H, W) where N is num_masks
                    mask_tensors = result.masks.data.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy() if hasattr(result, "boxes") and result.boxes is not None else np.ones(len(mask_tensors))

                    for mask_tensor, conf in zip(mask_tensors, confs):
                        # Ensure mask is scaled to original image size if needed
                        # YOLOv8 masks are typically scaled to original image dims
                        # if result.orig_shape is used, but verify.
                        mask_resized = cv2.resize(mask_tensor.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                        binary_mask = (mask_resized > 0.5).astype(np.uint8) # Binarize
                        poly = mask_to_polygon(binary_mask)
                        if poly is not None:
                            detected_panels.append({"poly": poly, "raw_conf": float(conf)})
                except Exception as e:
                    log("INFER_WARN", f"Mask extraction issue for {sample_id}: {e}")

            # Fallback to bounding boxes if no masks or mask extraction failed
            if not detected_panels and hasattr(result, "boxes") and result.boxes is not None and len(result.boxes.xyxy) > 0:
                try:
                    boxes = result.boxes.xyxy.cpu().numpy() # x1, y1, x2, y2 format
                    confs = result.boxes.conf.cpu().numpy()

                    for (x1, y1, x2, y2), conf in zip(boxes, confs):
                        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                        # Create a binary mask from the bounding box
                        mask_from_bbox = np.zeros((img_height, img_width), dtype=np.uint8)
                        # Ensure coordinates are within image bounds
                        cv2.rectangle(mask_from_bbox,
                                      (max(0, x1i), max(0, y1i)),
                                      (min(img_width-1, x2i), min(img_height-1, y2i)),
                                      1, -1) # Fill rectangle with 1
                        poly = mask_to_polygon(mask_from_bbox)
                        if poly is not None:
                            detected_panels.append({"poly": poly, "raw_conf": float(conf)})
                except Exception as e:
                    log("INFER_WARN", f"Bounding box fallback issue for {sample_id}: {e}")
        else:
            log("INFER", f"No detections found for sample {sample_id}.")


        log("INFER", f"Detected {len(detected_panels)} panel candidates for {sample_id}.")

        # 3. Quantify & Buffer Search
        # Center point of the image (in pixel coordinates) for buffer search
        image_center_px = Point(img_width / 2.0, img_height / 2.0)
        chosen_panel_info = None
        final_buffer_sqft = None

        m_per_px = meters_per_pixel(effective_lat, PipelineConfig.DEFAULT_ZOOM) # Meters per pixel for this image

        # Iterate through buffer sizes, smallest to largest, to find a panel within
        for buf_sqft in PipelineConfig.BUFFER_SEARCH_SQFT:
            radius_m = math.sqrt((buf_sqft * 0.092903) / math.pi) # Convert sqft to sqm, then to radius
            radius_px = radius_m / m_per_px

            # Create a shapely circle representing the buffer in pixel coordinates
            circle_buffer_px = image_center_px.buffer(radius_px)

            # Save buffer image for audit
            save_buffer_image(img_np, image_center_px, radius_px,
                              os.path.join(self.artifacts_dir, f"{sample_id}_buffer_{buf_sqft}sqft.png"))

            overlapping_panels = []
            for panel_candidate in detected_panels:
                try:
                    # Check for intersection between the detected panel polygon and the buffer circle
                    intersection_area = panel_candidate["poly"].intersection(circle_buffer_px).area
                    if intersection_area > 0:
                        overlapping_panels.append((intersection_area, panel_candidate))
                except Exception as e:
                    log("BUFFER_WARN", f"Error checking intersection for panel in {sample_id}: {e}")

            if overlapping_panels:
                # If multiple panels overlap, choose the one with the largest overlap area
                overlapping_panels.sort(key=lambda x: x[0], reverse=True)
                chosen_panel_info = overlapping_panels[0][1] # Get the panel dict
                final_buffer_sqft = buf_sqft
                log("BUFFER", f"Found PV in {buf_sqft} sqft buffer for {sample_id}.")
                break # Found a panel, stop searching larger buffers
            else:
                log("BUFFER", f"No PV in {buf_sqft} sqft buffer for {sample_id}.")

        # Prepare final record
        if chosen_panel_info is None:
            # No solar detected within any buffer
            return self._create_empty_record(
                sample_id, original_lat, original_lon, image_source,
                current_qc,
                ["no_pv_in_buffer"] + current_qc["flags"] # Combine reasons
            )

        # Solar panel detected and chosen
        pv_polygon = chosen_panel_info["poly"]
        raw_confidence = chosen_panel_info.get("raw_conf", 0.0)

        # Convert polygon area from pixels^2 to m^2
        pv_area_sqm = pv_polygon.area * (m_per_px ** 2)

        # Confidence calibration (unchanged)
        calibrated_confidence = calibrate_confidence(raw_confidence, pv_area_sqm)

        # Panel count MUST come from number of detected instances
        panel_count = len(detected_panels)

        # Capacity derived only from selected panel area
        capacity_kw = estimate_capacity_kw(pv_area_sqm)

        # 4. Explainability & QC Status Finalization
        # Determine final QC status based on image quality and calibrated confidence
        qc_status = "VERIFIABLE"
        has_solar_output = True
        if ("CLOUD_GLARE" in current_qc["flags"] or "DARK_SHADOW" in current_qc["flags"]) and calibrated_confidence < 0.75:
            # If image quality is poor AND model confidence is not high enough to override
            qc_status = "NOT_VERIFIABLE"
            has_solar_output = None # Cannot definitively say present or not present

        reason_codes = reason_codes_for(pv_polygon, calibrated_confidence, current_qc["flags"])

        # Save artifacts: mask image and audit overlay
        save_mask_image(img_np, pv_polygon, os.path.join(self.artifacts_dir, f"{sample_id}_mask.png"))

        audit_overlay = img_np.copy()
        single_poly_for_draw = ensure_single_polygon(pv_polygon)

        if single_poly_for_draw is not None:
            exterior_pts_for_draw = (
                np.array(single_poly_for_draw.exterior.coords)
                .astype(np.int32)
                .reshape(-1, 1, 2)
            )

            # draw filled polygon on a separate layer
            poly_layer = audit_overlay.copy()
            cv2.fillPoly(poly_layer, [exterior_pts_for_draw], (0, 255, 0))

            # manual alpha blending (OpenCV-safe)
            audit_overlay = cv2.addWeighted(
                poly_layer, 0.3,
                audit_overlay, 0.7,
                0
            )

            # draw outline
            cv2.polylines(
                audit_overlay,
                [exterior_pts_for_draw],
                True,
                (0, 128, 0),
                2
            )

        # Add text overlays to the audit image
        text_color_ok = (0, 255, 0) # Green
        text_color_warn = (0, 165, 255) # Orange
        text_color_err = (0, 0, 255) # Red

        cv2.putText(audit_overlay, f"QC: {qc_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    text_color_ok if qc_status == "VERIFIABLE" else text_color_err, 2)
        cv2.putText(audit_overlay, f"CONF: {calibrated_confidence:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    text_color_ok if calibrated_confidence > 0.7 else text_color_warn, 2)
        cv2.putText(audit_overlay, f"AREA: {pv_area_sqm:.1f} m²", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color_ok, 2)
        cv2.putText(audit_overlay, f"PANELS: {panel_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color_ok, 2)
        cv2.imwrite(os.path.join(self.artifacts_dir, f"{sample_id}_audit_overlay.jpg"), audit_overlay)
        log("ARTIFACTS", f"Saved mask and audit overlay for {sample_id}")


        # Final output record
        final_record = {
            "sample_id": sample_id,
            "lat": original_lat,
            "lon": original_lon,
            "has_solar": has_solar_output,
            "confidence": float(calibrated_confidence),
            "pv_area_sqm_est": round(pv_area_sqm, 3),
            "buffer_radius_sqft": int(final_buffer_sqft),
            "qc_status": qc_status,
            "reason_codes": reason_codes,
            "bbox_or_mask": encode_polygon_to_geojson_string(pv_polygon),
            "image_metadata": {
                "source": image_source,
                "capture_date": PipelineConfig.PROCESSING_DATE_PLACEHOLDER, # See NOTE in PipelineConfig
                "capacity_assumption_wp_per_m2": PipelineConfig.WP_PER_M2,
                "panel_area_assumption_m2": PipelineConfig.PANEL_AREA_M2, # Added for completeness
                "confidence_calibrated": True,
                "confidence_method": "area_weighted_heuristic" # More accurate description
            }
        }
        # Add optional fields requested in problem description, but not in mandatory output example
        # (panel_count_est, capacity_kw_est). These are useful.
        final_record["panel_count_est"] = panel_count
        final_record["capacity_kw_est"] = capacity_kw

        return final_record

# =========================================================
# ENTRYPOINT (main function for script execution)
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Rooftop PV verification pipeline - final")
    parser.add_argument("--input_file", required=True, help="Path to input CSV/XLSX with sample_id, latitude, longitude")
    parser.add_argument("--output_dir", required=True, help="Output folder for JSON results and artifacts")
    parser.add_argument("--model_path", required=True, help="Path to the YOLO segmentation model (.pt file)")
    parser.add_argument("--api_key", required=False, help="Google Static Maps API key (optional). If not provided, ESRI is primary.")
    args = parser.parse_args()

    # Retrieve API key from arguments or environment variable
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log("WARNING", "Google Static Maps API key not provided. Relying solely on ESRI World Imagery.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load input data
    log("SETUP", f"Loading input data from {args.input_file}")
    if args.input_file.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input_file)
    else:
        df = pd.read_csv(args.input_file)

    # Normalize column names to lowercase for robust access
    df.columns = [col.strip().lower() for col in df.columns]
    required_cols = {"sample_id", "latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {list(required_cols)}. Found: {list(df.columns)}")

    pipeline = SolarVerificationPipeline(args.model_path, args.output_dir, api_key)

    all_results = []
    for idx, row in df.iterrows():
        try:
            record = pipeline.process(row)
            all_results.append(record)
        except Exception as e:
            log("ERROR", f"Failed to process sample_id={row.get('sample_id', idx)}: {e}")
            # Create a minimal error record if an unhandled exception occurs
            all_results.append({
                "sample_id": row.get("sample_id", idx),
                "lat": row.get("latitude", None),
                "lon": row.get("longitude", None),
                "has_solar": None, # Undetermined due to error
                "confidence": 0.0,
                "pv_area_sqm_est": 0.0,
                "buffer_radius_sqft": None,
                "qc_status": "NOT_VERIFIABLE",
                "reason_codes": [f"pipeline_error: {str(e)}"],
                "bbox_or_mask": None,
                "image_metadata": {
                    "source": "N/A",
                    "capture_date": PipelineConfig.PROCESSING_DATE_PLACEHOLDER,
                    "capacity_assumption_wp_per_m2": PipelineConfig.WP_PER_M2,
                    "confidence_calibrated": False,
                    "confidence_method": "none"
                },
                "panel_count_est": 0,
                "capacity_kw_est": 0.0
            })

    # 5. Store Results
    output_json_path = os.path.join(args.output_dir, "final_predictions.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    log("DONE", f"Processed {len(all_results)} samples. Results saved to {output_json_path}")
    log("DONE", f"Audit artifacts are in {pipeline.artifacts_dir}")

if __name__ == "__main__":
    main()