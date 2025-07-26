import supervision as sv
from supervision.assets import download_assets, VideoAssets
import numpy as np
import cv2  # Import OpenCV for image saving
import os   # Import os for path manipulation

# SOURCE_VIDEO_PATH = download_assets(VideoAssets.VEHICLES)
SOURCE_VIDEO_PATH = "/home/geso/Tdetectors/models_src/MOT/vehicles_2.mp4"
from ultralytics import YOLO

# model = YOLO("/home/geso/Tdetectors/models/modelVehicle/runs/train/exp/weights/best.pt")
model = YOLO("yolo12n.pt")

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator and LabelAnnotator
# acquire first video frame
iterator = iter(generator)
frame = next(iterator)
# model prediction on single frame and conversion to supervision Detections
results = model(frame, verbose=False, iou=0.5, device="cuda:0")[0]

# convert to Detections
detections = sv.Detections.from_ultralytics(results)
# only consider class id from selected_classes define above
detections = detections

# format custom labels
labels = [
    f"{[class_id]} {confidence:0.2f}"
    for confidence, class_id in zip(detections.confidence, detections.class_id)
]

# settings
LINE_START = sv.Point(0 + 50, 1500)
LINE_END = sv.Point(3840 - 50, 1500)

TARGET_VIDEO_PATH = f"/home/geso/Tdetectors/models_src/MOT/result.mp4"

# create BYTETracker instance
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=25,
    minimum_consecutive_frames=3)

byte_tracker.reset()

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create LineZone instance, it is previously called LineCounter class
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create instance of BoxAnnotator, LabelAnnotator, and TraceAnnotator
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=20)

# create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=2)

# --- NEW: Define the folder to save crops ---
CROPS_FOLDER = "/home/geso/Tdetectors/models_src/MOT/cropped_objects"
os.makedirs(CROPS_FOLDER, exist_ok=True) # Create the folder if it doesn't exist

# define call back function to be used in video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=True, tracker="botsort.yaml")[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id}  {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    # --- NEW: Save crops ---
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x1 < x2 and y1 < y2: # Ensure valid bounding box
            cropped_image = frame[y1:y2, x1:x2]
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else "unknown"
            class_id = detections.class_id[i] if detections.class_id is not None else "unknown"

            # Create a unique filename for each crop
            # Example: frame_0001_track_123_class_0.png
            crop_filename = os.path.join(
                CROPS_FOLDER,
                f"frame_{index:05d}_track_{tracker_id}_class_{class_id}.png"
            )
            cv2.imwrite(crop_filename, cropped_image)
            # print(f"Saved crop: {crop_filename}") # Uncomment for debugging

    # update line counter
    line_zone.trigger(detections)
    # return frame with box and line annotated result
    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

# process the whole video
sv.process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)
print("we run finish")