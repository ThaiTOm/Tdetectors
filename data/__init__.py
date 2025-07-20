import os
import shutil
import sys
from linecache import cache

from omegaconf import OmegaConf
from roboflow import Roboflow
from tqdm import tqdm
import glob # For listing files more robustly

# Determine the parent directory
parent_dir = os.getenv("ROOT_PATH", r"/home/geso/Tdetectors")

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
# Load the global configuration file

cfg = OmegaConf.load(os.path.join(parent_dir, "config/global_config.yml"))

def dataset_download_local(cfg_dataset):
    rf = Roboflow(api_key=cfg.roboflow.api_key)
    project = rf.workspace(cfg_dataset.workspace).project(cfg_dataset.project)
    version = project.version(cfg_dataset.version)
    dataset = version.download(cfg_dataset.yolo_model)
    dataset_path = dataset.location
    print(f"Dataset downloaded to: {dataset_path}")
    return dataset

def merge_dataset(folder_path, output_path="Car-Counting-1"):
    # cp -r vehicle-1/train/* Car-Counting-1/train/
    for dt in tqdm(os.listdir(folder_path), desc="Merging datasets"):
        dt_path = os.path.join(folder_path, dt)
        if os.path.isdir(dt_path) and dt != output_path:
            print(f"Merging dataset: {dt}")
            for split in ["train", "valid", "test"]:
                src_split_path = os.path.join(dt_path, split)
                if os.path.exists(src_split_path):
                    dst_split_path = os.path.join(output_path, split)
                    if not os.path.exists(dst_split_path):
                        os.makedirs(dst_split_path)

                    # copy images and labels
                    for file_type in ["images", "labels"]:
                        src_file_path = os.path.join(src_split_path, file_type)
                        dst_file_path = os.path.join(dst_split_path, file_type)

                        if os.path.exists(src_file_path):
                            if not os.path.exists(dst_file_path):
                                os.makedirs(dst_file_path)
                            # Copy files from source to destination
                            for file_name in tqdm(os.listdir(src_file_path), desc=f"Copying {file_type} files"):
                                file_name = os.path.join(src_file_path, file_name)
                                shutil.copy(file_name, dst_file_path)


    return


def label_change(folder_path):
    for file_name in tqdm(os.listdir(folder_path), desc="Processing labels"):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            modified_lines = []
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if line.strip():  # Ensure the line is not empty or just whitespace
                    # Replace the first character with '0'
                    # Keep the rest of the line from the second character onwards
                    modified_line = '0' + line[1:]
                    modified_lines.append(modified_line)
                else:
                    # Keep empty lines as they are
                    modified_lines.append(line)

            # Write the modified content back to the same file (overwriting it)
            with open(file_path, 'w') as f:
                f.writelines(modified_lines)


import argparse
import os
import torch
from ultralytics import YOLO


def train_model(opt):
    """
    Main function to run YOLOv8 training.

    This function initializes a YOLO model, checks for GPU availability,
    and starts the training process based on the provided options.
    It can also optionally export the best model to ONNX format after training.
    """
    # --- 1. Device Check: Check for GPU and print status ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    if device == 'cpu':
        print("⚠️ WARNING: No GPU detected. Training on CPU will be extremely slow.")

    # --- 2. Model Initialization ---
    # Load a YOLO model. Can be a pretrained model like 'yolov8n.pt'
    # or a path to a previously trained model for resuming 'runs/train/exp1/weights/last.pt'
    print(f"🤖 Loading model: {opt.model}")
    model = YOLO(opt.model)

    # --- 3. Start Training ---
    print("\n" + "=" * 50)
    print("🚦 Starting YOLOv8 training...")
    print(f"💾 Dataset: {opt.data}")
    print(f"🎯 Epochs: {opt.epochs}")
    print(f"📦 Batch Size: {opt.batch}")
    print(f"🖼️ Image Size: {opt.imgsz}")
    print("=" * 50 + "\n")

    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        name=opt.name,
        project=opt.project,  # Organizes experiments into a 'project' folder
        exist_ok=True,  # Allow overwriting existing experiment folder
        device=device,
        cache=opt.cache,  # Cache images for faster training
    )

    print("\n" + "✅" * 20)
    print("🎉 Training finished successfully!")
    print(f"📈 Results saved to: {results.save_dir}")
    print("✅" * 20 + "\n")

    # --- 4. Optional: Export the best model ---
    if opt.export_onnx:
        # Find the path to the best performing model
        best_model_path = os.path.join(results.save_dir, 'weights/best.pt')
        if os.path.exists(best_model_path):
            print(f"📦 Exporting the best model ({best_model_path}) to ONNX format...")
            # Load the best model
            best_model = YOLO(best_model_path)
            # Export the model
            best_model.export(format='onnx', imgsz=opt.imgsz)
            print("✅ Export complete.")
        else:
            print(f"❌ Could not find best model at {best_model_path} for export.")


