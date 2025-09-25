from torch.utils.data.sampler import Sampler
import os
import cv2
import collections
from PIL import Image
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
import torch
import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import BatchSampler

class CombinedVehicleDataset(Dataset):
    """
    A combined PyTorch Dataset for multiple vehicle re-identification datasets,
    including VehicleID, Vehicle-1M, and the AICity22 Challenge.

    This class loads data from all sources, processes the AICity22 videos
    to extract and save vehicle crops, ensures vehicle IDs are unique across
    datasets by prefixing them, and creates a unified set of process IDs (PIDs)
    for training. It is compatible with RandomIdentitySampler.
    """

    def __init__(self,
                 # Paths for VehicleID dataset
                 vehicleID_list_path="/home/geso/Tdetectors/data/VehicleID/VehicleID_V1.0/train_test_split/train_list.txt",
                 vehicleID_root_dir="/home/geso/Tdetectors/data/VehicleID/VehicleID_V1.0/image",
                 # Paths for Vehicle-1M dataset
                 vehicle1M_txt_path="/home/geso/Tdetectors/data/Vehicle1M/Vehicle-1M/train-test-split/train_list.txt",
                 vehicle1M_root_dir="/home/geso/Tdetectors/data/Vehicle1M/Vehicle-1M/image",
                 # Paths for the AICity22 dataset
                 aicity_root_path="/home/geso/Tdetectors/data/aiCity22",
                 aicity_crop_dir: str = "/home/geso/Tdetectors/data/aiCity22/crops",
                 aicity_split: str = 'train',
                 is_train=True,
                 transform=None):

        self.transform = transform
        self.is_train = is_train

        # Load data from each source
        vehicleID_data = self._load_vehicleID_data(vehicleID_list_path, vehicleID_root_dir)
        vehicle1M_data = self._load_vehicle1M_data(vehicle1M_txt_path, vehicle1M_root_dir)
        aicity_data = self._load_aicity_data(aicity_root_path, aicity_crop_dir, aicity_split)

        # Combine the data
        all_image_paths = (
                vehicleID_data['paths'] +
                vehicle1M_data['paths'] +
                aicity_data['paths'])

        all_labels = (
                [f"vehicleid_{l}" for l in vehicleID_data['labels']] +
                [f"vehicle1m_{l}" for l in vehicle1M_data['labels']] +
                [f"aicity_{l}" for l in aicity_data['labels']]
        )

        all_cams = (
                vehicleID_data['cams'] +
                vehicle1M_data['cams'] +
                aicity_data['cams'])

        print("-" * 40)
        print(f"Loaded {len(vehicleID_data['paths'])} images from VehicleID.")
        print(f"Loaded {len(vehicle1M_data['paths'])} images from Vehicle-1M.")
        print(f"Loaded {len(aicity_data['paths'])} images from AICity22.")
        print(f"Total images in combined dataset: {len(all_image_paths)}")
        print("-" * 40)

        self.image_paths = all_image_paths
        self.cams = all_cams

        if self.is_train:
            unique_labels = sorted(list(set(all_labels)))
            self.label_to_pid = {label: pid for pid, label in enumerate(unique_labels)}
            self.pids = [self.label_to_pid[label] for label in all_labels]
            self.num_classes = len(unique_labels)  ### <<< MODIFIED ###
            print(f"Total number of unique vehicle IDs (classes) for training: {self.num_classes}")
        else:
            self.pids = all_labels
            self.num_classes = -1  # Not applicable for testing in this context

        self.data_info = self.image_paths

    def get_class(self, index: int) -> int:
        """
        Returns the PID (class ID) for the image at the given index.
        This is required by the RandomIdentitySampler.
        """
        return self.pids[index]

    def _load_vehicleID_data(self, list_path, root_dir):
        """
        Helper to load data from the VehicleID dataset.
        Parses a list file (e.g., train_list.txt) which contains image names and vehicle IDs.
        """
        paths, labels, cams = [], [], []
        if not list_path or not root_dir:
            return {'paths': [], 'labels': [], 'cams': []}

        try:
            with open(list_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    # The format is typically "image_name.jpg vehicle_id"
                    img_name, v_id = line.strip().split()
                    full_path = os.path.join(root_dir, img_name + ".jpg")

                    if os.path.exists(full_path):
                        paths.append(full_path)
                        labels.append(v_id)
                        # VehicleID dataset does not provide camera IDs. Use a placeholder.
                        cams.append(-1)
        except Exception as e:
            print(f"Error loading VehicleID dataset from {list_path}: {e}")

        return {'paths': paths, 'labels': labels, 'cams': cams}

    # ... (rest of the helper methods _load_vehicle1M_data, _load_aicity_data, etc. remain unchanged) ...
    # ... I am omitting them here for brevity, but they should be kept in your final code. ...

    def _load_vehicle1M_data(self, txt_path, root_dir):
        """Helper to load data from the Vehicle-1M text file."""
        paths, labels, cams = [], [], []
        if not txt_path or not root_dir: return {'paths': [], 'labels': [], 'cams': []}
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    img_path, v_id, c_id = parts[0], parts[1], parts[2]
                    full_path = os.path.join(root_dir, img_path)
                    if os.path.exists(full_path):
                        paths.append(full_path)
                        labels.append(v_id)
                        cams.append(int(c_id))
        except Exception as e:
            print(f"Error loading Vehicle-1M dataset from {txt_path}: {e}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    def _load_aicity_data(self, root_dir: str, crop_dir: str, split: str) -> Dict:
        """
        Helper to load data from the AICity22 Challenge dataset.
        It extracts vehicle crops from video files and saves them to disk for caching.
        """
        root_path = Path(root_dir)
        crop_path = Path(crop_dir)
        paths, labels, cams = [], [], []

        # FAST PATH: Load pre-extracted crops if they exist
        if crop_path.exists() and any(crop_path.iterdir()):
            print(f"INFO: Loading pre-extracted AICity crops from cache: {crop_path}")
            for img_file in sorted(list(crop_path.glob("*.jpg"))):
                try:
                    # Filename format: c<cam_id>_v<vehicle_id>_f<frame_num>.jpg
                    parts = img_file.stem.split('_')
                    cam_id_str = parts[0]
                    vehicle_id_str = parts[1]

                    paths.append(str(img_file))
                    labels.append(vehicle_id_str.replace('v', ''))
                    cams.append(int(cam_id_str.replace('c', '')))
                except (ValueError, IndexError):
                    print(f"WARNING: Skipping malformed filename in cache: {img_file.name}")

            if paths:
                return {'paths': paths, 'labels': labels, 'cams': cams}

        # SLOW PATH: Extract crops from videos for the first time
        print(f"INFO: Cache directory '{crop_path}' is empty or not found.")
        print("      Proceeding with SLOW PATH: extracting vehicle crops from videos.")
        print("      This is a one-time process and may take a very long time.")
        crop_path.mkdir(parents=True, exist_ok=True)
        print(f"      Created cache directory at: {crop_path.resolve()}")

        vehicle_data = self._group_aicity_vehicle_ids(root_path, split=split)
        total_sightings = sum(len(s) for s in vehicle_data.values())
        processed_count = 0
        video_caps = {}  # Cache open video captures to avoid re-opening files

        try:
            for vehicle_id, sightings in vehicle_data.items():
                for sighting in sightings:
                    video_file = sighting['video_path']
                    frame_num, bbox = sighting['frame_num'], sighting['bbox']

                    try:
                        cam_name = Path(video_file).parent.name
                        cam_id = int(cam_name.replace('c', ''))
                    except (ValueError, IndexError):
                        continue

                    if video_file not in video_caps:
                        cap = cv2.VideoCapture(video_file)
                        if not cap.isOpened(): continue
                        video_caps[video_file] = cap

                    cap = video_caps[video_file]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                    ret, frame = cap.read()

                    if ret:
                        x, y, w, h = bbox
                        if w <= 0 or h <= 0: continue
                        crop = frame[max(0, y):y + h, max(0, x):x + w]
                        if crop.size == 0: continue

                        filename = f"c{cam_id:03d}_v{vehicle_id:05d}_f{frame_num:06d}.jpg"
                        save_path = crop_path / filename
                        cv2.imwrite(str(save_path), crop)

                        paths.append(str(save_path))
                        labels.append(str(vehicle_id))
                        cams.append(cam_id)

                    processed_count += 1
                    if processed_count % 2000 == 0:
                        print(f"      ... processed {processed_count} / {total_sightings} sightings.")
        finally:
            for cap in video_caps.values():
                cap.release()

        print(f"INFO: AICity crop extraction complete. Saved {len(paths)} images to {crop_path}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    @staticmethod
    def _group_aicity_vehicle_ids(dataset_path: Path, split: str) -> Dict[int, List[Dict]]:
        """Parses the AIC22 dataset to group all vehicle sightings by their unique ID for a given split."""
        list_cam_path = dataset_path / 'list_cam.txt'
        if not list_cam_path.exists():
            raise FileNotFoundError(f"Required file 'list_cam.txt' not found in {dataset_path}")

        cam_paths = []
        with open(list_cam_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(f'./{split}/'):
                    cam_paths.append(dataset_path / line.lstrip('./'))

        if not cam_paths:
            print(f"WARNING: No camera paths found for split '{split}' in {list_cam_path}.")
            return {}

        vehicle_data = collections.defaultdict(list)
        for cam_path in cam_paths:
            gt_path = cam_path / 'gt/gt.txt'
            video_path = cam_path / 'vdo.avi'
            if not gt_path.exists() or not video_path.exists():
                continue

            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6: continue
                    try:
                        frame, v_id, x, y, w, h = map(int, parts[:6])
                        if v_id == -1: continue
                        vehicle_data[v_id].append({
                            'video_path': str(video_path),
                            'frame_num': frame,
                            'bbox': [x, y, w, h]
                        })
                    except ValueError:
                        continue

        print(f"INFO: Found {len(vehicle_data)} unique vehicle IDs in '{split}' split from GT files.")
        return dict(vehicle_data)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        pid = self.pids[idx]
        camid = self.cams[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {img_path}. {e}. Returning a placeholder.")
            if self.transform:
                blank_pil = Image.new('RGB', (128, 256), (0, 0, 0))
                return self.transform(blank_pil), -1, -1, 0
            placeholder_tensor = torch.zeros(3, 256, 128)
            return placeholder_tensor, -1, -1, 0

        if self.transform:
            image = self.transform(image)

        pid_val = np.int64(pid) if self.is_train else pid
        camid_val = np.int64(camid)

        return image, pid_val, camid_val, 0


def train_collate_fn(batch):
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index in range(len(self.data_source.data_info)):
            pid = self.data_source.get_class(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length



class DistributedRandomIdentityBatchSampler(BatchSampler):
    """
    A distributed-aware batch sampler that randomly samples N identities, and for
    each identity, randomly samples K instances to form a batch of size N*K.
    This sampler is designed to be used with DistributedDataParallel.

    Args:
    - data_source (Dataset): The dataset (or Subset) to sample from. Must have a .pids attribute.
    - batch_size (int): Number of examples in a batch.
    - num_instances (int): Number of instances per identity in a batch.
    - world_size (int): Total number of processes for distributed training.
    - rank (int): Rank of the current process.
    - epoch (int): The current epoch number (used for shuffling).
    """

    def __init__(self, data_source, batch_size, num_instances, world_size, rank, epoch=0):
        # The __init__ of BatchSampler requires a sampler. We use a dummy one.
        super().__init__(torch.utils.data.sampler.SequentialSampler(data_source), batch_size, drop_last=False)

        self.data_source = data_source
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.world_size = world_size
        self.rank = rank
        self.epoch = epoch

        self.index_dic = defaultdict(list)
        # This loop correctly creates a map of {pid: [subset_idx1, subset_idx2, ...]}
        for index, pid in enumerate(self.data_source.pids):
            self.index_dic[pid].append(index)

        self.pids = sorted(list(self.index_dic.keys()))
        self.num_pids = len(self.pids)

        if self.num_pids_per_batch * self.world_size > self.num_pids:
            raise ValueError(f"Not enough PIDs to create unique batches for all processes.")

    def __iter__(self):
        # ### CORRECTED LOGIC ###
        # 1. Get shuffled INDICES
        g = torch.Generator()
        g.manual_seed(self.epoch)
        shuffled_indices = torch.randperm(self.num_pids, generator=g).tolist()

        # 2. Use the indices to shuffle the actual PIDs
        shuffled_pids = [self.pids[i] for i in shuffled_indices]

        # 3. Split the shuffled PIDs among ranks
        pids_for_this_rank = shuffled_pids[self.rank::self.world_size]

        batch_idxs_dict = defaultdict(list)

        # 4. Pre-process indices for each PID this rank is responsible for
        for pid in pids_for_this_rank:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                # This line will now work because `idxs` is never empty
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()

            random.shuffle(idxs)

            for i in range(0, len(idxs) - self.num_instances + 1, self.num_instances):
                batch_idxs_dict[pid].append(idxs[i:i + self.num_instances])

        avai_pids = pids_for_this_rank.copy()
        final_batches = []

        # 5. Create batches until we run out of PIDs
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)

            current_batch = []
            for pid in selected_pids:
                batch_for_pid = batch_idxs_dict[pid].pop(0)
                current_batch.extend(batch_for_pid)

                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

            final_batches.append(current_batch)

        random.shuffle(final_batches)
        return iter(final_batches)

    def __len__(self):
        pids_for_this_rank = len(self.pids[self.rank::self.world_size])
        total_chunks = 0
        for pid in self.pids[self.rank::self.world_size]:
            num_images = len(self.index_dic[pid])
            total_chunks += num_images // self.num_instances
        return total_chunks // self.num_pids_per_batch

    def set_epoch(self, epoch):
        self.epoch = epoch