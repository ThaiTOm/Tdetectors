import cv2
import collections
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Union


def group_vehicle_ids(dataset_path: Union[str, Path], split: str) -> Dict[int, List[Dict]]:
    """
    Parses the AIC22 dataset to group all vehicle sightings by their unique ID.

    This function is now aware of the dataset's structure:
    - For 'train' and 'validation' splits, it uses the ground truth from 'gt/gt.txt'.
    - For the 'test' split, it uses the tracking results from 'mtsc/mtsc_*.txt'.
    - It uses 'list_cam.txt' to determine the correct camera folders for each split.

    Args:
        dataset_path: The root path to the AIC22 CityFlowV2 dataset.
        split: The data split to process ('train', 'validation', or 'test').

    Returns:
        A dictionary where keys are integer vehicle IDs and values are lists.
        Each item in the list is a dictionary representing a single sighting:
        {
            'video_path': Path to the source video file,
            'frame_num': The frame number of the sighting,
            'bbox': A list [x, y, width, height] for the bounding box.
        }
    """
    root_path = Path(dataset_path)
    if not root_path.is_dir():
        raise FileNotFoundError(f"The dataset root path does not exist: {root_path}")

    list_cam_path = root_path / 'list_cam.txt'
    if not list_cam_path.exists():
        raise FileNotFoundError(f"Required file 'list_cam.txt' not found in {root_path}")

    # --- 1. Get the list of camera paths for the specified split ---
    cam_paths_to_process = []
    with open(list_cam_path, 'r') as f:
        for line in f:
            line = line.strip()
            # The paths in the file look like './train/S01/c001/'
            if line.startswith(f'./{split}/'):
                # Construct the full, absolute-style path from the root
                cam_paths_to_process.append(root_path / line.lstrip('./'))

    if not cam_paths_to_process:
        print(f"WARNING: No camera paths found for split '{split}' in list_cam.txt.")
        return {}

    print(f"INFO: Found {len(cam_paths_to_process)} camera directories for the '{split}' split.")

    # --- 2. Process each camera path to extract vehicle data ---
    vehicle_data = collections.defaultdict(list)

    for cam_path in cam_paths_to_process:
        source_files = []
        # --- 3. Determine the data source based on the split ---
        if split in ['train', 'validation']:
            source_files = list(cam_path.glob('gt/gt.txt'))
            print(f"INFO: [Train/Validation] Processing {source_files[0] if source_files else 'N/A'}")
        elif split == 'test':
            source_files = sorted(list(cam_path.glob('mtsc/mtsc_*.txt')))
            print(f"INFO: [Test] Processing {len(source_files)} MTSC files in {cam_path / 'mtsc'}")

        video_path = cam_path / 'vdo.avi'
        if not video_path.exists():
            print(f"WARNING: Video file not found at {video_path}, skipping this camera.")
            continue

        for data_file_path in source_files:
            with open(data_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue

                    # Format is the same for gt.txt and mtsc_*.txt:
                    # [frame, ID, left, top, width, height, ...]
                    try:
                        frame_num = int(parts[0])
                        vehicle_id = int(parts[1])
                        # Ignore placeholder IDs like -1 often found in detection files
                        if vehicle_id == -1:
                            continue
                        x = int(parts[2])
                        y = int(parts[3])
                        w = int(parts[4])
                        h = int(parts[5])
                    except ValueError:
                        continue  # Skip malformed lines

                    sighting_info = {
                        'video_path': str(video_path),
                        'frame_num': frame_num,
                        'bbox': [x, y, w, h]
                    }
                    vehicle_data[vehicle_id].append(sighting_info)

    print(f"\nINFO: Processing complete. Found {len(vehicle_data)} unique vehicle IDs in the '{split}' split.")
    return dict(vehicle_data)


# The display function remains the same as it's independent of the data source
def display_vehicle_sightings(all_vehicles_data: Dict, vehicle_id_to_show: int):
    """
    Extracts and displays all cropped images for a specific vehicle ID.
    (This function does not need any changes)
    """
    if vehicle_id_to_show not in all_vehicles_data:
        print(f"ERROR: Vehicle ID {vehicle_id_to_show} not found in the dataset.")
        return

    sightings = all_vehicles_data[vehicle_id_to_show]
    num_sightings = len(sightings)
    print(f"INFO: Found {num_sightings} sightings for Vehicle ID {vehicle_id_to_show}.")

    # ... (rest of the function is identical to the previous version) ...
    cols = min(num_sightings, 5)
    rows = (num_sightings + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_sightings > 1 else [axes]

    for i, sighting in enumerate(sightings):
        video_path, frame_num, bbox = sighting['video_path'], sighting['frame_num'], sighting['bbox']
        x, y, w, h = bbox
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        if ret:
            vehicle_crop = frame[y:y + h, x:x + w]
            ax = axes[i]
            ax.imshow(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB))
            cam_name = Path(video_path).parent.name
            ax.set_title(f"Cam: {cam_name}\nFrame: {frame_num}")
            ax.axis('off')

    for j in range(num_sightings, len(axes)): axes[j].axis('off')
    plt.suptitle(f"All Sightings for Vehicle ID: {vehicle_id_to_show}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- HOW TO USE ---

if __name__ == '__main__':
    # 1. Set the path to the root of your AIC22 dataset
    DATASET_ROOT_PATH = "."  # <--- IMPORTANT: CHANGE THIS PATH

    # --- EXAMPLE 1: Process the 'train' split (uses gt/gt.txt) ---
    print("--- Parsing 'train' split to group vehicle IDs ---")
    train_vehicles = group_vehicle_ids(DATASET_ROOT_PATH, split='validation')
    if train_vehicles:
        vehicle_id_to_visualize = 96  # Pick a known ID from the training set
        print(f"\n--- Displaying all images for Vehicle ID: {vehicle_id_to_visualize} from 'train' split ---")
        display_vehicle_sightings(train_vehicles, vehicle_id_to_visualize)

    print("\n" + "=" * 80 + "\n")

    # # --- EXAMPLE 2: Process the 'test' split (uses mtsc/*.txt) ---
    # print("--- Parsing 'test' split to group vehicle IDs ---")
    # test_vehicles = group_vehicle_ids(DATASET_ROOT_PATH, split='test')
    # if test_vehicles:
    #     # We don't know the test IDs, so let's just pick the first one we find
    #     first_test_id = list(test_vehicles.keys())[0]
    #     print(f"\n--- Displaying all images for Vehicle ID: {first_test_id} from 'test' split ---")
    #     display_vehicle_sightings(test_vehicles, first_test_id)