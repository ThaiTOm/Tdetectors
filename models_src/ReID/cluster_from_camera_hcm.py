import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import shutil
from processor import get_model  # Assume get_model is in processor.py
import yaml
import imagehash
from datetime import datetime, timedelta
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

# --- HELPER FUNCTIONS (Re-ID and Data Management) ---
# Most functions from the previous script are kept, with minor adjustments if needed.

def get_images_from_first_hour(base_folder, duration_hours=1.0):
    """
    STEP 0: Scans the base folder, finds the earliest capture, and returns all
    image paths within the specified duration from that start time.
    """
    print(f"\n--- STEP 0: Collecting Source Images from First {duration_hours} Hour(s) ---")
    # ... (This function remains unchanged) ...
    print(f"Scanning data source: '{base_folder}'")

    if not os.path.exists(base_folder):
        print(f"Error: The directory '{base_folder}' does not exist.")
        return []

    timestamp_folders = []
    timestamp_format = '%Y-%m-%d_%H-%M-%S'

    for folder_name in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, folder_name)):
            try:
                ts = datetime.strptime(folder_name, timestamp_format)
                timestamp_folders.append((ts, os.path.join(base_folder, folder_name)))
            except ValueError:
                continue

    if not timestamp_folders:
        print("Error: No valid timestamped subfolders found in the directory.")
        return []

    timestamp_folders.sort(key=lambda x: x[0])
    start_time, end_time = timestamp_folders[0][0], timestamp_folders[0][0] + timedelta(hours=duration_hours)

    print(f"First capture found at: {start_time.strftime(timestamp_format)}")
    print(f"Processing data up to:  {end_time.strftime(timestamp_format)}")

    image_paths_to_process = []
    for ts, folder_path in timestamp_folders:
        if start_time <= ts < end_time:
            image_paths_to_process.extend(glob.glob(os.path.join(folder_path, '*.jpg')))

    print(f"Collected {len(image_paths_to_process)} total source images for detection.")
    return image_paths_to_process


# --- NEW CORE FUNCTION: YOLO DETECTION ---

def _save_crop(args):
    """Helper function to save a single cropped image. Used for parallel execution."""
    cropped_obj, save_path = args
    try:
        cropped_obj.save(save_path, quality=95)
        return save_path
    except Exception as e:
        print(f"Warning: Could not save cropped image to '{save_path}'. Error: {e}")
        return None

def run_yolo_detection_and_crop(yolo_model,
                                     source_image_paths,
                                     output_dir,
                                     batch_size=16,
                                     confidence_threshold=0.5,
                                     target_classes=None):
    """
    Optimized version that runs YOLO detection in batches and saves crops in parallel.
    """
    if target_classes is None:
        target_classes = ["car", "truck", "bus"] # Default vehicle classes

    print(f"\n--- STEP 1: Detecting and Cropping Objects (Optimized) ---")
    if os.path.exists(output_dir):
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    target_class_ids = {
        k for k, v in yolo_model.names.items() if v in target_classes
    }

    crops_to_save = []

    for i in tqdm(range(0, len(source_image_paths), batch_size), desc="Running YOLO Batch Prediction"):
        batch_paths = source_image_paths[i:i + batch_size]
        batch_images, valid_paths = [], []
        for path in batch_paths:
            try:
                batch_images.append(Image.open(path).convert('RGB'))
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Cannot load '{path}'. Skipping. Error: {e}")

        if not batch_images: continue

        results = yolo_model.predict(batch_images, verbose=False, conf=confidence_threshold)

        for original_image, img_path, result in zip(batch_images, valid_paths, results):
            for box in result.boxes:
                if box.cls.item() in target_class_ids:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cropped_obj = original_image.crop(xyxy)
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    obj_id = f"{int(xyxy[0])}_{int(xyxy[1])}"
                    save_path = os.path.join(output_dir, f"{base_name}_obj_{obj_id}.jpg")
                    crops_to_save.append((cropped_obj, save_path))

    print(f"\nFound {len(crops_to_save)} objects. Saving cropped images in parallel...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(_save_crop, crops_to_save), total=len(crops_to_save), desc="Saving Crops"))

    cropped_image_paths = [path for path in results if path is not None]

    print(f"\nDetection and cropping complete. Successfully saved {len(cropped_image_paths)} objects.")
    return cropped_image_paths


# --- ### MODIFIED ###: OPTIMIZED DEDUPLICATION FUNCTION ---

def _compute_hash(args):
    """Helper function to compute hash for a single image. For parallel execution."""
    img_path, hash_size = args
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            return img_path, imagehash.dhash(img, hash_size=hash_size)
    except Exception as e:
        # print(f"Warning: Could not compute hash for '{img_path}'. Skipping. Error: {e}")
        return None

def deduplicate_from_path_list(image_paths_list, target_dir, hash_size=8, hamming_distance_threshold=5):
    """
    STEP 2: Takes a list of cropped object paths and performs hash-based deduplication.
    This version is optimized by first calculating all hashes in parallel.
    """
    print(f"\n--- STEP 2: Initial Deduplication of Cropped Objects (Hash Level) ---")
    if not image_paths_list:
        print("Warning: Received an empty list of images for deduplication.")
        return []

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # --- Part 1: Compute all hashes in parallel (I/O bound) ---
    print(f"Step 2.1: Calculating hashes for {len(image_paths_list)} images in parallel...")
    path_hash_list = []
    args_list = [(path, hash_size) for path in image_paths_list]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(_compute_hash, args_list), total=len(args_list), desc="Hashing Images"))

    # Filter out any images that failed to process
    path_hash_list = [res for res in results if res is not None]
    if not path_hash_list:
        print("Could not generate hashes for any of the images. Exiting deduplication.")
        return []

    # --- Part 2: Sequentially compare hashes to find unique set (CPU bound) ---
    print("Step 2.2: Finding unique images by comparing hashes...")
    unique_hashes_paths = []
    deduplicated_paths = []

    for img_path, current_hash in tqdm(path_hash_list, desc="Deduplicating Objects"):
        is_duplicate = any(
            (current_hash - existing_hash) <= hamming_distance_threshold
            for _, existing_hash in unique_hashes_paths
        )
        if not is_duplicate:
            unique_hashes_paths.append((img_path, current_hash))
            dest_path = os.path.join(target_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
            deduplicated_paths.append(dest_path)

    print(f"Kept {len(deduplicated_paths)} unique objects after hash deduplication.")
    return deduplicated_paths


# --- RE-ID PIPELINE FUNCTIONS (Unchanged) ---

def load_reid_model(model_path, device):
    try:
        with open("./config/config.yaml", "r") as stream:
            data = yaml.safe_load(stream)
    except FileNotFoundError:
        print("Error: Re-ID config file './config/config.yaml' not found."); exit()
    model = get_model(data, device)
    state_dict = torch.load(model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device);
    model.eval()
    print(f"Re-ID model loaded from {model_path} and set to evaluation mode.")
    return model

def get_image_transforms(): return transforms.Compose(
    [transforms.Resize((256, 256), antialias=True), transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            with Image.open(path).convert('RGB') as img:
                if self.transform:
                    img = self.transform(img)
                return img, path
        except Exception:
            return None, path

def collate_fn_skip_corrupted(batch):
    batch = [(img, path) for img, path in batch if img is not None]
    if not batch: return None, None
    images, paths = zip(*batch)
    return torch.stack(images, 0), list(paths)

@torch.no_grad()
def extract_features(model, image_paths, transform, device, batch_size=64):
    print(f"\n--- STEP 3: Extracting Features (Optimized) ---")
    if not image_paths: return torch.tensor([]), []
    try:
        model = torch.compile(model)
        print("Model compiled successfully with torch.compile().")
    except Exception as e:
        print(f"Could not compile model with torch.compile(): {e}. Continuing without it.")

    dataset = ImagePathDataset(image_paths, transform)
    num_workers = os.cpu_count() // 2
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_skip_corrupted, num_workers=num_workers, pin_memory=True, drop_last=False)
    model.eval(); model.to(device)
    all_features, all_processed_paths = [], []

    for batch_tensor, batch_paths in tqdm(dataloader, desc="Extracting Features"):
        if batch_tensor is None:
            print(f"Warning: A full batch of images was corrupted. Skipping.")
            continue
        batch_tensor = batch_tensor.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            _, _, ffs, _ = model(batch_tensor, cam=None, view=None)
            if not ffs:
                print(f"Warning: Model returned no features for a batch. First path: {batch_paths[0]}")
                continue
            batch_features = nn.functional.normalize(torch.cat(ffs, dim=1), p=2, dim=1)
        all_features.append(batch_features.cpu())
        all_processed_paths.extend(batch_paths)

    if not all_features: return torch.tensor([]), []
    return torch.cat(all_features, dim=0), all_processed_paths

def cluster_images(features, image_paths, distance_threshold=1.0):
    print(f"\n--- STEP 4: Clustering Objects based on Re-ID Features ---")
    if features.shape[0] == 0: return {}, []
    print(f"Clustering {len(image_paths)} objects with distance threshold: {distance_threshold}")
    clustering = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='average', distance_threshold=distance_threshold)
    labels = clustering.fit_predict(features.numpy())
    grouped_images = {label: [] for label in np.unique(labels)}
    for img_path, label in zip(image_paths, labels): grouped_images[label].append(img_path)
    print(f"Found {len(grouped_images)} unique object clusters.")
    return grouped_images, labels

def select_top_k_representatives(grouped_images, k=20):
    """
    Selects top K clusters based on the number of images in each cluster.
    """
    print(f"\n--- STEP 5: Selecting Top {k} Most Frequent Objects ---")
    if not grouped_images: return [], {}
    # Sort clusters by the number of images they contain, in descending order
    sorted_clusters = sorted(grouped_images.items(), key=lambda item: len(item[1]), reverse=True)
    top_k_clusters = sorted_clusters[:k]
    # The representative is the first image in each of the top clusters
    representative_paths = [paths[0] for _, paths in top_k_clusters if paths]
    top_k_grouped_images = {label: paths for label, paths in top_k_clusters}
    print(f"Selected {len(representative_paths)} representative images from the most frequent clusters.")
    return representative_paths, top_k_grouped_images

def save_final_results(representative_paths, output_dir):
    print(f"\n--- STEP 6: Saving Final Results ---")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Copying {len(representative_paths)} final representative images to '{output_dir}'...")
    for img_path in tqdm(representative_paths, desc="Saving Final Images"): shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
    print(f"Process complete! Final unique images are saved in '{output_dir}'")

def save_top_clusters_for_inspection(top_k_grouped_images, output_dir):
    if not top_k_grouped_images: return
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    print(f"\nSaving top clusters for inspection to '{output_dir}'...")
    for label, paths in tqdm(top_k_grouped_images.items(), desc="Saving Inspection Groups"):
        group_dir = os.path.join(output_dir, f"group_{label}_size_{len(paths)}");
        os.makedirs(group_dir, exist_ok=True)
        for img_path in paths: shutil.copy(img_path, os.path.join(group_dir, os.path.basename(img_path)))
    print(f"Inspection files saved.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- 1. CONFIGURE YOUR PARAMETERS HERE ---

    # --- ### MODIFIED ###: Workflow Control ---
    # Set to True to skip detection and hash-deduplication, using existing images from OLD_IMAGES_DIR.
    # This assumes the images in OLD_IMAGES_DIR are the ones you want to run Re-ID on directly.
    USE_OLD_IMAGES = True
    # Directory to use if USE_OLD_IMAGES is True.
    # This is typically the output of a previous run's detection step (`CROPPED_OBJECTS_TEMP_DIR`)
    OLD_IMAGES_DIR = "temp_cropped_objects"

    # --- Data and Time ---
    CAPTURED_DATA_FOLDER = '/home/geso/Tdetectors/data/cameraHCM/captured_images'
    PROCESS_DURATION_HOURS = 0.1
    TOP_K_CLUSTERS_TO_SAVE = 20

    # --- Model Paths ---
    YOLO_MODEL_PATH = "yolo12m"
    REID_MODEL_PATH = "/home/geso/Tdetectors/models_src/ReID/logs/VRIC/Baseline/14/last.pt"

    # --- Detection & Clustering Parameters ---
    DETECTION_CONFIDENCE_THRESHOLD = 0.7
    HAMMING_THRESHOLD = 10
    DISTANCE_THRESHOLD_REID = 0.5

    # --- Output Folder Names ---
    CROPPED_OBJECTS_TEMP_DIR = "temp_cropped_objects"
    DEDUPLICATED_HASH_TEMP_DIR = "temp_deduplicated_hash"
    FINAL_OUTPUT_DIR = "final_top_20_reid_objects"
    INSPECTION_OUTPUT_DIR = "inspection_top_20_reid_clusters"

    # --- 2. SCRIPT EXECUTION ---

    # Check paths
    for path in [REID_MODEL_PATH]:
        if not os.path.exists(path):
            print(f"FATAL ERROR: Re-ID model path not found -> '{path}'"); exit()
    if not USE_OLD_IMAGES and not os.path.exists(CAPTURED_DATA_FOLDER):
        print(f"FATAL ERROR: Source data path not found -> '{CAPTURED_DATA_FOLDER}'"); exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- ### MODIFIED ###: Main pipeline logic with conditional deduplication ---
    images_for_reid = []  # This list will hold the images for the Re-ID step
    source_image_count = 0
    total_detected_objects = 0

    if USE_OLD_IMAGES:
        print(f"\n--- WORKFLOW: Using pre-existing images from '{OLD_IMAGES_DIR}' ---")
        print("--- SKIPPING Detection and Hash Deduplication as per 'USE_OLD_IMAGES=True' ---")

        if not os.path.exists(OLD_IMAGES_DIR):
            print(f"FATAL ERROR: The specified directory for old images does not exist: '{OLD_IMAGES_DIR}'"); exit()

        # Use glob to find all common image types and assign them for Re-ID processing
        images_for_reid = glob.glob(os.path.join(OLD_IMAGES_DIR, '*.jpg')) + \
                          glob.glob(os.path.join(OLD_IMAGES_DIR, '*.png'))

        if not images_for_reid:
            print(f"No images found in '{OLD_IMAGES_DIR}'. Exiting."); exit()


        total_detected_objects = len(images_for_reid) # For summary purposes
        print(f"Found {len(images_for_reid)} images to process for Re-ID.")

    else:
        print("\n--- WORKFLOW: Running full Detection -> Deduplication -> Re-ID pipeline ---")
        # Step 0: Get source image paths from the first hour
        source_image_paths = get_images_from_first_hour(CAPTURED_DATA_FOLDER, PROCESS_DURATION_HOURS)
        source_image_count = len(source_image_paths)
        if not source_image_paths: print("No source images found. Exiting."); exit()

        # Step 1: Run YOLO detection and crop objects
        yolo_model = YOLO(YOLO_MODEL_PATH)
        images_for_deduplication = run_yolo_detection_and_crop(
            yolo_model, source_image_paths, CROPPED_OBJECTS_TEMP_DIR,
            confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,
            target_classes=["car", "truck", "bus"]
        )
        total_detected_objects = len(images_for_deduplication)
        if not images_for_deduplication: print("YOLO found no objects of interest. Exiting."); exit()

        # Step 2: Deduplicate cropped objects using image hash (only in this workflow)
        images_for_reid = deduplicate_from_path_list(
            image_paths_list=images_for_deduplication,
            target_dir=DEDUPLICATED_HASH_TEMP_DIR,
            hash_size=8,
            hamming_distance_threshold=HAMMING_THRESHOLD
        )
        if not images_for_reid: print("No unique objects remained after hash deduplication. Exiting."); exit()

    # --- The rest of the pipeline now uses `images_for_reid` for both workflows ---


    # Step 3: Extract Re-ID features
    reid_model = load_reid_model(REID_MODEL_PATH, device)
    transform = get_image_transforms()
    features, processed_paths = extract_features(reid_model, images_for_reid, transform, device)
    if features.shape[0] == 0: print("Re-ID feature extraction failed. Exiting."); exit()

    # Step 4: Cluster objects based on Re-ID features
    all_grouped_images, _ = cluster_images(features, processed_paths, DISTANCE_THRESHOLD_REID)

    # Step 5: Select top K most frequent objects (This function is already correct)
    top_k_representatives, top_k_groups_for_inspection = select_top_k_representatives(all_grouped_images, k=TOP_K_CLUSTERS_TO_SAVE)

    # Step 6 & 7: Save final and inspection results
    save_final_results(top_k_representatives, FINAL_OUTPUT_DIR)
    save_top_clusters_for_inspection(top_k_groups_for_inspection, INSPECTION_OUTPUT_DIR)

    # --- 3. FINAL SUMMARY (Updated) ---
    print("\n" + "=" * 50)
    print("--- FULL PIPELINE SUMMARY ---")
    if USE_OLD_IMAGES:
        print(f"Workflow: Used pre-existing images (Deduplication skipped).")
        print(f"Images loaded for Re-ID from '{OLD_IMAGES_DIR}': {len(images_for_reid)}")
    else:
        print(f"Workflow: Full detection and deduplication pipeline.")
        print(f"Source images processed: {source_image_count}")
        print(f"Total objects detected by YOLO: {total_detected_objects}")
        print(f"Objects remaining for Re-ID (after hash deduplication): {len(images_for_reid)}")

    # Common stats for both workflows
    print("-" * 50)
    print(f"Unique object clusters found by Re-ID: {len(all_grouped_images)}")
    print(f"Saved top {len(top_k_representatives)} most frequent objects.")
    print(f"\nFINAL RESULTS saved in: '{FINAL_OUTPUT_DIR}'")
    print(f"INSPECTION FILES saved in: '{INSPECTION_OUTPUT_DIR}'")
    print("=" * 50)