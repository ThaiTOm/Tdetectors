import os
import random
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# ==========================================================================================
# 1. DATASET CLASS (Simplified version from your script)
#    We will load the raw PIL images, without any transformations, to see them clearly.
# ==========================================================================================

class SimpleVehicleDataset(Dataset):
    """
    A minimal dataset class to load image paths and labels.
    It returns the raw PIL image, which is perfect for visualization.
    """

    def __init__(self,
                 vehicleID_list_path="/home/geso/Tdetectors/data/VehicleID/VehicleID_V1.0/train_test_split/train_list.txt",
                 vehicleID_root_dir="/home/geso/Tdetectors/data/VehicleID/VehicleID_V1.0/image"):

        # Load the data using your existing logic
        vehicleID_data = self._load_vehicleID_data(vehicleID_list_path, vehicleID_root_dir)

        self.image_paths = vehicleID_data['paths']
        # For simplicity, we'll just use the original string labels from the file
        self.pids = vehicleID_data['labels']

        print("-" * 40)
        print(f"Loaded {len(self.image_paths)} images from VehicleID.")
        print("-" * 40)

    def _load_vehicleID_data(self, list_path, root_dir):
        paths, labels = [], []
        with open(list_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                img_name, v_id = line.strip().split()
                full_path = os.path.join(root_dir, img_name + ".jpg")
                if os.path.exists(full_path):
                    paths.append(full_path)
                    labels.append(v_id)  # Keep the original string ID
        return {'paths': paths, 'labels': labels}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        pid = self.pids[idx]
        # Open the image as a PIL image
        image = Image.open(img_path).convert('RGB')
        return image, pid


# ==========================================================================================
# 2. HELPER FUNCTIONS TO FIND AND DISPLAY PAIRS
# ==========================================================================================

def display_pair(image1, pid1, image2, pid2, main_title):
    """Displays a pair of images side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Image 1
    axes[0].imshow(image1)
    axes[0].set_title(f"Vehicle ID: {pid1}")
    axes[0].axis('off')

    # Image 2
    axes[1].imshow(image2)
    axes[1].set_title(f"Vehicle ID: {pid2}")
    axes[1].axis('off')

    fig.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()


def preprocess_data(dataset):
    """
    Creates a dictionary mapping each PID to a list of its image indices.
    This is a very efficient way to look up images by their ID.
    """
    print("Preprocessing dataset to map IDs to images...")
    pid_to_indices = defaultdict(list)
    for idx, (img, pid) in enumerate(dataset):
        pid_to_indices[pid].append(idx)
    print("Preprocessing complete.")
    return pid_to_indices


# ==========================================================================================
# 3. MAIN EXECUTION BLOCK
# ==========================================================================================

if __name__ == '__main__':
    # --- IMPORTANT: Make sure these paths are correct for your system ---
    VEHICLE_ID_LIST_PATH = "/home/geso/Tdetectors/data/VehicleID/VehicleID_V1.0/train_test_split/train_list.txt"
    VEHICLE_ID_ROOT_DIR = "/home/geso/Tdetectors/data/VehicleID/VehicleID_V1.0/image"

    # 1. Load the dataset
    dataset = SimpleVehicleDataset(
        vehicleID_list_path=VEHICLE_ID_LIST_PATH,
        vehicleID_root_dir=VEHICLE_ID_ROOT_DIR
    )

    # 2. Create the PID -> [indices] mapping for efficient lookup
    pid_map = preprocess_data(dataset)

    # --- Find and Display a Positive Pair (Same ID) ---
    print("\nFinding a positive pair (same Vehicle ID)...")

    # Find PIDs that have at least two images
    pids_with_multiple_images = [pid for pid, indices in pid_map.items() if len(indices) >= 2]

    if pids_with_multiple_images:
        # Pick a random PID from this list
        target_pid = random.choice(pids_with_multiple_images)

        # Pick two random, different indices for that PID
        index1, index2 = random.sample(pid_map[target_pid], 2)

        # Retrieve the images
        image1, pid1 = dataset[index1]
        image2, pid2 = dataset[index2]

        # Display them
        display_pair(image1, pid1, image2, pid2, "Positive Pair (Same Vehicle ID)")
    else:
        print("Could not find any Vehicle IDs with more than one image to form a positive pair.")

    # --- Find and Display a Negative Pair (Different IDs) ---
    print("\nFinding a negative pair (different Vehicle IDs)...")

    if len(pid_map) >= 2:
        # Pick two different random PIDs
        pid1_key, pid2_key = random.sample(list(pid_map.keys()), 2)

        # Pick one random index from each PID's list
        index1 = random.choice(pid_map[pid1_key])
        index2 = random.choice(pid_map[pid2_key])

        # Retrieve the images
        image1, pid1 = dataset[index1]
        image2, pid2 = dataset[index2]

        # Display them
        display_pair(image1, pid1, image2, pid2, "Negative Pair (Different Vehicle IDs)")
    else:
        print("Could not find two different Vehicle IDs to form a negative pair.")