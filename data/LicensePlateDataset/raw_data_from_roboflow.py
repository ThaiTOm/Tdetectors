import os
import sys
from omegaconf import OmegaConf
from roboflow import Roboflow
from tqdm import tqdm
# Determine the parent directory
parent_dir = os.getenv("ROOT_PATH", r"/home/geso/Tdetectors")

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
# Load the global configuration file
from data import *

cfg = OmegaConf.load(os.path.join(parent_dir, "config/global_config.yml"))

def cc_pintel_download_local():
    cfg_path = cfg.roboflow.license_plate_w8chc
    dataset = dataset_download_local(cfg_path)

    cfg_path = cfg.roboflow.license_plate_7zrdm
    dataset = dataset_download_local(cfg_path)

    return dataset

def process_raw_data():
    # This function is intended to process the raw data downloaded from Roboflow.
    # Implement the processing logic here. -> convert all label to vehicle only
    folder_data_path = os.path.join(parent_dir, "data/LicensePlateDataset/license-plate-1")
    train_path = os.path.join(folder_data_path, "train", "labels")
    valid_path = os.path.join(folder_data_path, "valid", "labels")
    test_path = os.path.join(folder_data_path, "test", "labels")

    print(f"Processing raw data in {train_path} and {valid_path}")
    label_change(train_path)
    label_change(valid_path)
    label_change(test_path)

    return

def push_dataset_to_roboflow():
    # This function is intended to push the processed dataset back to Roboflow.
    # Implement the upload logic here.
    API_KEY = cfg.roboflow.api_key_upload # Replace with your actual API Key
    WORKSPACE_URL = "finetune1"  # e.g., "my-awesome-project-123"
    PROJECT_URL = "detectors-license-plate"  # e.g., "vehicles-detection"

    DATA_FOLDER_PATH = "/home/geso/Tdetectors/data/LicensePlateDataset/License-Plate-1"

    UPLOAD_SPLIT = None  # Set to 'train' if you're only uploading training data

    # --- Initialize Roboflow ---
    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace(WORKSPACE_URL)
    project = workspace.project(PROJECT_URL)

    print(f"Connected to Roboflow project: {project.name}")

    # --- Uploading Data ---

    # If your data is already split into train/valid/test subfolders within DATA_FOLDER_PATH,
    # you can iterate and upload each split.
    # This assumes structure like: DATA_FOLDER_PATH/train/images, DATA_FOLDER_PATH/train/labels
    splits = ["train", "valid", "test"]  # Or just the ones you have

    for split_name in splits:
        split_images_path = os.path.join(DATA_FOLDER_PATH, split_name, "images")
        split_labels_path = os.path.join(DATA_FOLDER_PATH, split_name, "labels")

        if os.path.exists(split_images_path) and os.path.exists(split_labels_path):
            print(f"\nProcessing {split_name} split from: {split_images_path} and {split_labels_path}")

            image_files = glob.glob(os.path.join(split_images_path, "*"))  # Get all image files

            if not image_files:
                print(f"No image files found in {split_images_path}. Skipping this split.")
                continue

            for image_path in image_files:
                try:
                    # Construct the corresponding label file path
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    label_path = os.path.join(split_labels_path, base_filename + ".txt")

                    if not os.path.exists(label_path):
                        print(f"Warning: No label file found for {image_path}. Skipping image.")
                        continue

                    print(f"Uploading {os.path.basename(image_path)} to '{split_name}' split...")

                    # The core upload call for a single image + annotation pair
                    project.upload(
                        image_path=image_path,
                        annotation_path=label_path,
                        split=split_name,
                        # Optional: Set to True to overwrite if an image with the same name exists
                        # Be careful with this, especially if you want to add new images only.
                        # Default is False (won't upload if image with same name exists)
                        overwrite=False,
                        # If using YOLO format, specify model_format='yolov5' or 'yolov8'
                        # Roboflow's YOLO format expects: class_id x_center y_center width height
                        model_format="yolov12",
                    )
                    print(f"Successfully uploaded {os.path.basename(image_path)}.")

                except Exception as e:
                    print(f"Error uploading {os.path.basename(image_path)}: {e}")
        else:
            print(
                f"Skipping {split_name} split: image or label directories not found at {split_images_path} and {split_labels_path}")

    print("\nUpload process complete. Check your Roboflow project dashboard.")

push_dataset_to_roboflow()