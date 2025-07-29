import cv2
import os
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
# --- MODIFIED: Keep TextRecognition import if you use it, otherwise PaddleOCR is sufficient ---
from paddleocr import PaddleOCR, TextRecognition

# --- Your Custom Configuration Loading ---
import sys
from omegaconf import OmegaConf
from tqdm import tqdm
from ultralytics.utils.ops import non_max_suppression

# Determine the parent directory
parent_dir = os.getenv("ROOT_PATH", os.path.expanduser("~/Tdetectors"))
sys.path.insert(0, parent_dir)

try:
    cfg = OmegaConf.load(os.path.join(parent_dir, "config/global_config.yml"))
except FileNotFoundError:
    print(f"Error: Configuration file not found at {os.path.join(parent_dir, 'config/global_config.yml')}")
    sys.exit(1)

# Assuming crop_license_plates is defined later, which is fine
from utils import *

# --- Configuration ---
CAR_MODEL_PATH = cfg.MODEL_CHECKPOINT.car
LP_MODEL_PATH = cfg.MODEL_CHECKPOINT.license_plate
CAR_CONF_THRESHOLD = 0.7
LP_CONF_THRESHOLD = 0.8

# --- MODIFIED: Global Dictionary to store OCR results with scores ---
# This will hold {tracker_id: (ocr_text, score)}
tracked_plates_data = {}


# --- Main Pipeline Functions (Unchanged) ---
def run_detection_pipeline(frame: np.ndarray, car_model: YOLO, lp_model: YOLO) -> (sv.Detections, sv.Detections):
    """
    Runs car and license plate detection on a single frame.
    Returns car_detections (untracked) and lp_detections (in absolute coordinates).
    """
    car_results = car_model.predict(frame, conf=CAR_CONF_THRESHOLD, verbose=False, iou=0.2)[0]
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
            for lp_box, lp_conf, lp_class in zip(lp_detections_in_crop.xyxy, lp_detections_in_crop.confidence,
                                                 lp_detections_in_crop.class_id):
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


# --- process_image (Unchanged, as it's for single frames) ---
def process_image(source_path: str, target_path: str, car_model: YOLO, lp_model: YOLO, paddle_ocr: PaddleOCR):
    """Processes a single image."""
    print(f"Processing image: {source_path}")
    frame = cv2.imread(source_path)
    if frame is None:
        print(f"Error: Could not read image from {source_path}")
        return

    car_detections, lp_detections = run_detection_pipeline(frame, car_model, lp_model)
    cropped_plates = crop_license_plates(frame, lp_detections)

    # In single image mode, we just get the best text without tracking
    ocr_results_with_scores = LicensePlateReader(paddle_ocr, cropped_plates)
    ocr_texts = [text for text, score in ocr_results_with_scores]

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
        print(
            f"Warning: Mismatch in OCR texts ({len(ocr_texts)}) and LP detections ({len(lp_detections)}). Skipping LP labels.")

    cv2.imwrite(target_path, annotated_frame)
    print(f"Result saved to: {target_path}")

    if len(lp_detections) > 0 and len(ocr_texts) > 0:
        for i, ocr_text in enumerate(ocr_texts):
            print(f"Detected LP (Image): {ocr_text}")


def run_detection_and_ocr_pipeline(
    frame: np.ndarray,
    lp_model: YOLO,
    modelLicensePlateRecognizer,
    tracked_cars: sv.Detections
) -> (list, sv.Detections):
    """
    Detects license plates and runs OCR within the boxes of tracked cars.
    This ensures a direct and correct association between a plate and a tracker ID.
    """
    ocr_pipeline_results = []
    all_lp_xyxy = []
    all_lp_confidence = []
    # --- FIX: Initialize a list to hold class IDs ---
    all_lp_class_id = []

    if tracked_cars.tracker_id is None:
        return [], sv.Detections.empty()

    for car_box, tracker_id in zip(tracked_cars.xyxy, tracked_cars.tracker_id):
        x1_car, y1_car, x2_car, y2_car = map(int, car_box)
        car_crop = frame[y1_car:y2_car, x1_car:x2_car]

        if car_crop.size == 0:
            continue

        lp_results = lp_model(car_crop, conf=LP_CONF_THRESHOLD, verbose=False)[0]

        if len(lp_results.boxes) > 0:
            lp_detections_in_crop = sv.Detections.from_ultralytics(lp_results)
            lp_crops = crop_license_plates(car_crop, lp_detections_in_crop)
            ocr_results_with_scores = LicensePlateReader(modelLicensePlateRecognizer, lp_crops)

            for i, (text, score) in enumerate(ocr_results_with_scores):
                if text:
                    ocr_pipeline_results.append({
                        'tracker_id': tracker_id,
                        'text': text,
                        'score': score
                    })

                lp_box_relative = lp_detections_in_crop.xyxy[i]
                x1_lp_abs = int(x1_car + lp_box_relative[0])
                y1_lp_abs = int(y1_car + lp_box_relative[1])
                x2_lp_abs = int(x1_car + lp_box_relative[2]) # Corrected typo: should be x1_car
                y2_lp_abs = int(y1_car + lp_box_relative[3]) # Corrected typo: should be y1_car

                all_lp_xyxy.append([x1_lp_abs, y1_lp_abs, x2_lp_abs, y2_lp_abs])
                all_lp_confidence.append(lp_detections_in_crop.confidence[i])
                # --- FIX: Append the class ID for the current license plate ---
                all_lp_class_id.append(lp_detections_in_crop.class_id[i])


    all_lp_detections = sv.Detections(
        xyxy=np.array(all_lp_xyxy),
        confidence=np.array(all_lp_confidence),
        class_id=np.array(all_lp_class_id)
    ) if all_lp_xyxy else sv.Detections.empty()

    return ocr_pipeline_results, all_lp_detections


# --- THIS IS THE CORRECTED process_video FUNCTION ---
def process_video(source_path: str, target_path: str, car_model: YOLO, lp_model: YOLO, paddle_ocr: PaddleOCR):
    """Processes a video file, updating LP text if a better OCR score is found."""
    print(f"Processing video: {source_path}")
    video_info = sv.VideoInfo.from_video_path(source_path)
    byte_tracker = sv.ByteTrack(frame_rate=video_info.fps)

    # Annotators (no change)
    car_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
    car_label_annotator = sv.LabelAnnotator(color=sv.Color.BLUE, text_color=sv.Color.WHITE, text_scale=0.5)
    lp_box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
    lp_label_annotator = sv.LabelAnnotator(color=sv.Color.GREEN, text_color=sv.Color.WHITE, text_scale=0.7)

    # Clear previous OCR results for a new video
    global tracked_plates_data
    tracked_plates_data = {}

    def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
        # --- Step 1: Car Detection and Tracking ---
        car_results = car_model.predict(frame, conf=CAR_CONF_THRESHOLD, verbose=False, iou=0.2)[0]
        car_detections = sv.Detections.from_ultralytics(car_results)
        tracked_cars = byte_tracker.update_with_detections(car_detections)

        # --- Step 2: Integrated LP Detection and OCR ---
        # This function handles the association correctly and returns all data needed.
        ocr_pipeline_results, all_lp_detections = run_detection_and_ocr_pipeline(
            frame, lp_model, paddle_ocr, tracked_cars
        )

        # --- Step 3: Update Best Score (Now much simpler and correct) ---
        for result in ocr_pipeline_results:
            tracker_id = result['tracker_id']
            current_text = result['text']
            current_score = result['score']

            # The core logic is the same, but now it's guaranteed to be for the correct ID
            if tracker_id not in tracked_plates_data:
                tracked_plates_data[tracker_id] = (current_text, current_score)
            else:
                _, old_score = tracked_plates_data[tracker_id]
                if current_score > old_score:
                    tracked_plates_data[tracker_id] = (current_text, current_score)

        # --- Step 4: Annotation ---
        annotated_frame = frame.copy()

        # Annotate cars with the best-known OCR info
        car_labels = []
        if tracked_cars.tracker_id is not None:
            for tracker_id in tracked_cars.tracker_id:
                ocr_text, _ = tracked_plates_data.get(tracker_id, ("N/A", 0.0))
                car_labels.append(f"ID:{tracker_id} LP:{ocr_text}")

        # annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=tracked_cars)
        annotated_frame = car_box_annotator.annotate(scene=annotated_frame, detections=tracked_cars)
        annotated_frame = car_label_annotator.annotate(scene=annotated_frame, detections=tracked_cars, labels=car_labels)

        # Annotate all detected license plates
        if len(all_lp_detections) > 0:
            # For simplicity, we won't label individual LPs in this version,
            # as the main label is on the car. We just draw the boxes.
             annotated_frame = lp_box_annotator.annotate(scene=annotated_frame, detections=all_lp_detections)

        return annotated_frame

    # Manual loop with tqdm (no change)
    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    with sv.VideoSink(target_path, video_info) as sink:
        for frame_index, frame in enumerate(
                tqdm(frame_generator, total=video_info.total_frames, desc="Processing video")):
            processed_frame = callback(frame, frame_index)
            sink.write_frame(frame=processed_frame)

    print(f"Result saved to: {target_path}")

    # Save final results to a file (no change)
    output_ocr_filename = os.path.join(os.path.dirname(target_path),
                                       f"{os.path.splitext(os.path.basename(target_path))[0]}_ocr_results.txt")
    with open(output_ocr_filename, 'w') as f:
        if not tracked_plates_data:
            f.write("No valid license plates were tracked for this video.\n")
        else:
            f.write("Final tracked license plates (best score):\n")
            for car_id, (ocr_text, ocr_score) in sorted(tracked_plates_data.items()):
                f.write(f"ID: {car_id}, License Plate: {ocr_text}, Confidence Score: {ocr_score:.4f}\n")
    print(f"Final OCR results saved to: {output_ocr_filename}")


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

        from paddleocr import PaddleOCR

        # paddle_ocr = PaddleOCR(
        #     use_doc_orientation_classify=True,
        #     use_doc_unwarping=False,
        #     use_textline_orientation=True)

        from fast_plate_ocr import LicensePlateRecognizer

        modelLicensePlateRecognizer = LicensePlateRecognizer('cct-s-v1-global-model')

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
        process_image(source_path, output_path, car_model, lp_model, modelLicensePlateRecognizer)
    elif source_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(source_path, output_path, car_model, lp_model, modelLicensePlateRecognizer)
    else:
        print(f"Error: Unsupported file format '{source_extension}'.")