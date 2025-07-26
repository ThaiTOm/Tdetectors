import cv2
import os
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
from paddleocr import PaddleOCR, TextRecognition # Keep TextRecognition for clarity

# --- Your Custom Configuration Loading ---
import sys
from omegaconf import OmegaConf

# Determine the parent directory
parent_dir = os.getenv("ROOT_PATH", os.path.expanduser("~/Tdetectors"))
sys.path.insert(0, parent_dir)

try:
    cfg = OmegaConf.load(os.path.join(parent_dir, "config/global_config.yml"))
except FileNotFoundError:
    print(f"Error: Configuration file not found at {os.path.join(parent_dir, 'config/global_config.yml')}")
    sys.exit(1)

from utils import * # Assuming crop_license_plates and run_ocr_on_plates are here

# --- Configuration ---
CAR_MODEL_PATH = cfg.MODEL_CHECKPOINT.car
LP_MODEL_PATH = cfg.MODEL_CHECKPOINT.license_plate
CAR_CONF_THRESHOLD = 0.4
LP_CONF_THRESHOLD = 0.5

# --- Global Dictionary to store OCR results ---
# This will hold {tracker_id: FIRST_VALID_ocr_text_for_that_id}
detected_license_plates_ocr = {}

# --- Main Pipeline Functions (Unchanged) ---
# ... run_detection_pipeline and process_image are the same as before ...
def run_detection_pipeline(frame: np.ndarray, car_model: YOLO, lp_model: YOLO) -> (sv.Detections, sv.Detections):
    """
    Runs car and license plate detection on a single frame.
    Returns car_detections (untracked) and lp_detections (in absolute coordinates).
    """
    car_results = car_model(frame, conf=CAR_CONF_THRESHOLD, verbose=False)[0]
    car_detections = sv.Detections.from_ultralytics(car_results)

    all_lp_xyxy, all_lp_confidence, all_lp_class_id = [], [], []

    # Iterate through detected cars to find license plates within them
    for car_box in car_detections.xyxy:
        x1_car, y1_car, x2_car, y2_car = map(int, car_box)
        car_crop = frame[y1_car:y2_car, x1_car:x2_car]

        if car_crop.size == 0 or car_crop.shape[0] == 0 or car_crop.shape[1] == 0:
            continue

        lp_results = lp_model(car_crop, conf=LP_CONF_THRESHOLD, verbose=False)[0]

        if len(lp_results.boxes) > 0:
            lp_detections_in_crop = sv.Detections.from_ultralytics(lp_results)
            for lp_box, lp_conf, lp_class in zip(lp_detections_in_crop.xyxy, lp_detections_in_crop.confidence, lp_detections_in_crop.class_id):
                x1_lp_rel, y1_lp_rel, x2_lp_rel, y2_lp_rel = map(int, lp_box)
                x1_lp_abs, y1_lp_abs = x1_car + x1_lp_rel, y1_car + y1_lp_rel
                x2_lp_abs, y2_lp_abs = x1_car + x2_lp_rel, y1_car + y2_lp_rel

                all_lp_xyxy.append([x1_lp_abs, y1_lp_abs, x2_lp_abs, y2_lp_abs])
                all_lp_confidence.append(lp_conf)
                all_lp_class_id.append(lp_class)

    lp_detections = sv.Detections(
        xyxy=np.array(all_lp_xyxy),
        confidence=np.array(all_lp_confidence),
        class_id=np.array(all_lp_class_id)
    ) if len(all_lp_xyxy) > 0 else sv.Detections.empty()

    return car_detections, lp_detections

def process_image(source_path: str, target_path: str, car_model: YOLO, lp_model: YOLO, paddle_ocr: TextRecognition):
    """Processes a single image."""
    print(f"Processing image: {source_path}")
    frame = cv2.imread(source_path)
    if frame is None:
        print(f"Error: Could not read image from {source_path}")
        return

    car_detections, lp_detections = run_detection_pipeline(frame, car_model, lp_model)
    cropped_plates = crop_license_plates(frame, lp_detections)
    ocr_texts = run_ocr_on_plates(paddle_ocr, cropped_plates)

    car_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
    car_label_annotator = sv.LabelAnnotator(color=sv.Color.BLUE, text_color=sv.Color.WHITE)
    lp_box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
    lp_label_annotator = sv.LabelAnnotator(color=sv.Color.GREEN, text_color=sv.Color.WHITE, text_scale=0.7)

    annotated_frame = frame.copy()
    annotated_frame = car_box_annotator.annotate(scene=annotated_frame, detections=car_detections)
    car_labels = [f"Car {confidence:0.2f}" for confidence in car_detections.confidence]
    annotated_frame = car_label_annotator.annotate(scene=annotated_frame, detections=car_detections, labels=car_labels)
    annotated_frame = lp_box_annotator.annotate(scene=annotated_frame, detections=lp_detections)
    if len(ocr_texts) == len(lp_detections):
        annotated_frame = lp_label_annotator.annotate(scene=annotated_frame, detections=lp_detections, labels=ocr_texts)
    else:
        print(f"Warning: Mismatch in OCR texts ({len(ocr_texts)}) and LP detections ({len(lp_detections)}). Skipping LP labels.")

    cv2.imwrite(target_path, annotated_frame)
    print(f"Result saved to: {target_path}")

    if len(lp_detections) > 0 and len(ocr_texts) > 0:
        for i, ocr_text in enumerate(ocr_texts):
            print(f"Detected LP (Image): {ocr_text}")

# --- process_video with the "Lock-In" Logic ---

def process_video(source_path: str, target_path: str, car_model: YOLO, lp_model: YOLO, paddle_ocr: TextRecognition):
    """Processes a video file."""
    print(f"Processing video: {source_path}")
    video_info = sv.VideoInfo.from_video_path(source_path)
    byte_tracker = sv.ByteTrack(frame_rate=video_info.fps)

    car_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
    car_label_annotator = sv.LabelAnnotator(color=sv.Color.BLUE, text_color=sv.Color.WHITE, text_scale=0.5)
    trace_annotator = sv.TraceAnnotator(color=sv.Color.BLUE, thickness=2, trace_length=30)

    lp_box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
    lp_label_annotator = sv.LabelAnnotator(color=sv.Color.GREEN, text_color=sv.Color.WHITE, text_scale=0.7)

    # Clear previous OCR results for a new video
    global detected_license_plates_ocr
    detected_license_plates_ocr = {}

    def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
        car_detections, lp_detections = run_detection_pipeline(frame, car_model, lp_model)
        tracked_cars = byte_tracker.update_with_detections(car_detections)
        cropped_plates = crop_license_plates(frame, lp_detections)
        ocr_results = run_ocr_on_plates(paddle_ocr, cropped_plates)

        lp_labels_for_annotation = [""] * len(lp_detections)

        for i_car, car_box in enumerate(tracked_cars.xyxy):
            car_id = tracked_cars.tracker_id[i_car]
            x1_car, y1_car, x2_car, y2_car = map(int, car_box)

            for i_lp, lp_box in enumerate(lp_detections.xyxy):
                x1_lp, y1_lp, x2_lp, y2_lp = map(int, lp_box)

                x_overlap = max(0, min(x2_car, x2_lp) - max(x1_car, x1_lp))
                y_overlap = max(0, min(y2_car, y2_lp) - max(y1_car, y1_lp))
                lp_width = x2_lp - x1_lp
                lp_height = y2_lp - y1_lp

                if lp_width > 0 and lp_height > 0:
                    overlap_ratio_x = x_overlap / lp_width
                    overlap_ratio_y = y_overlap / lp_height

                    if overlap_ratio_x >= 0.8 and overlap_ratio_y >= 0.8:
                        # --- MODIFIED LOCK-IN LOGIC ---
                        current_ocr_text = ocr_results[i_lp].strip()

                        # 1. Check if the current OCR is valid (not an empty string).
                        # 2. Check if this car_id does NOT already have a plate assigned.
                        if current_ocr_text and car_id not in detected_license_plates_ocr:
                            # This is the FIRST valid OCR for this ID. Lock it in!
                            detected_license_plates_ocr[car_id] = current_ocr_text

                        # Set the annotation label for the LP box using the stored value
                        # This ensures it always shows the locked-in value, or N/A if none exists yet.
                        lp_labels_for_annotation[i_lp] = detected_license_plates_ocr.get(car_id, "N/A")
                        # --- END MODIFIED LOGIC ---

        # --- ANNOTATION (Now uses the locked-in values) ---
        annotated_frame = frame.copy()

        # Annotate cars with the stored (locked-in) OCR info
        car_labels = []
        for tracker_id in tracked_cars.tracker_id:
            # .get() will fetch the locked-in plate or default to "N/A"
            ocr_info = detected_license_plates_ocr.get(tracker_id, "N/A")
            car_labels.append(f"ID:{tracker_id} LP:{ocr_info}")

        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=tracked_cars)
        annotated_frame = car_box_annotator.annotate(scene=annotated_frame, detections=tracked_cars)
        annotated_frame = car_label_annotator.annotate(scene=annotated_frame, detections=tracked_cars, labels=car_labels)

        # Annotate license plates using the collected labels (which are also based on the locked-in value)
        if len(lp_labels_for_annotation) == len(lp_detections):
            annotated_frame = lp_box_annotator.annotate(scene=annotated_frame, detections=lp_detections)
            annotated_frame = lp_label_annotator.annotate(scene=annotated_frame, detections=lp_detections, labels=lp_labels_for_annotation)
        else:
            print(f"Warning: Mismatch in LP labels and detections. Skipping LP labels.")

        return annotated_frame

    sv.process_video(source_path=source_path, target_path=target_path, callback=callback)
    print(f"Result saved to: {target_path}")

    # --- Save detected license plates to a file after video processing ---
    # This part does not need to change, as it correctly reads the final locked-in values.
    output_ocr_filename = os.path.join(os.path.dirname(target_path), f"{os.path.splitext(os.path.basename(target_path))[0]}_ocr_results.txt")
    with open(output_ocr_filename, 'w') as f:
        if not detected_license_plates_ocr:
            f.write("No valid license plates were locked in for this video.\n")
        else:
            # Sort by ID for consistent output
            for car_id, ocr_text in sorted(detected_license_plates_ocr.items()):
                f.write(f"ID: {car_id}, License Plate: {ocr_text}\n")
    print(f"OCR results saved to: {output_ocr_filename}")


# --- Main execution block (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run car and license plate detection with OCR.")
    parser.add_argument("--source", default="/home/geso/Tdetectors/models/MOT/vehicles_2.mp4", help="Path to the source image or video file.")
    parser.add_argument("--car-model", default=CAR_MODEL_PATH, help="Path to car detection model. Overrides config.")
    parser.add_argument("--lp-model", default=LP_MODEL_PATH, help="Path to license plate detection model. Overrides config.")

    args = parser.parse_args()

    print("Loading YOLO models...")
    try:
        car_model = YOLO(args.car_model)
        lp_model = YOLO(args.lp_model)
    except Exception as e:
        print(f"Error loading YOLO models: {e}"); exit()

    print("Loading PaddleOCR model...")
    try:
        paddle_ocr = TextRecognition()
    except Exception as e:
        print(f"Error loading PaddleOCR model: {e}"); exit()

    source_path = args.source
    _, source_extension = os.path.splitext(source_path)
    source_extension = source_extension.lower()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(source_path)
    output_path = os.path.join(output_dir, base_name)

    if source_extension in ['.jpg', '.jpeg', '.png']:
        process_image(source_path, output_path, car_model, lp_model, paddle_ocr)
    elif source_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(source_path, output_path, car_model, lp_model, paddle_ocr)
    else:
        print(f"Error: Unsupported file format '{source_extension}'.")