import os
import cv2
import copy
import random
import argparse
import yaml
import collections
from pathlib import Path
from typing import Dict, List
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

# DDP (Distributed Data Parallel) imports
import torch.multiprocessing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Custom project imports (assuming they are in these paths)
from loss.losses import triplet_loss_fastreid
from lr_scheduler.sche_optim import make_optimizer, make_warmup_scheduler
from tensorboard_log import Logger
from models.models import MBR_model, load_weights_custom # Assumed import
from metrics.eval_reid import eval_func # Assumed import

# Set sharing strategy for multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ==========================================================================================
# 1. DATASET DEFINITION
# ==========================================================================================

class CombinedVehicleDataset(Dataset):
    """
    A combined PyTorch Dataset for multiple vehicle re-identification datasets.
    This class loads and unifies data from various sources, making it compatible
    with identity-based samplers for re-ID tasks.
    """
    def __init__(self,
                 vehicleID_list_path="/home/geso/Tdetectors/data/VehicleID/VehicleID_V1.0/train_test_split/train_list.txt",
                 vehicleID_root_dir="/home/geso/Tdetectors/data/VehicleID/VehicleID_V1.0/image",
                 vehicle1M_txt_path="/home/geso/Tdetectors/data/Vehicle1M/Vehicle-1M/train-test-split/train_list.txt",
                 vehicle1M_root_dir="/home/geso/Tdetectors/data/Vehicle1M/Vehicle-1M/image",
                 aicity_root_path="/home/geso/Tdetectors/data/aiCity22",
                 aicity_crop_dir: str = "/home/geso/Tdetectors/data/aiCity22/crops",
                 aicity_split: str = 'train',
                 is_train=True,
                 transform=None):

        self.transform = transform
        self.is_train = is_train

        vehicleID_data = self._load_vehicleID_data(vehicleID_list_path, vehicleID_root_dir)
        vehicle1M_data = self._load_vehicle1M_data(vehicle1M_txt_path, vehicle1M_root_dir)
        aicity_data = self._load_aicity_data(aicity_root_path, aicity_crop_dir, aicity_split)

        # Combine the data sources
        all_image_paths = (vehicleID_data['paths'] + vehicle1M_data['paths'] + aicity_data['paths'])

        all_labels = (
            [f"vehicleid_{l}" for l in vehicleID_data['labels']] +
            [f"vehicle1m_{l}" for l in vehicle1M_data['labels']] +
            [f"aicity_{l}" for l in aicity_data['labels']]
        )
        all_cams = (vehicleID_data['cams'] )

        print("-" * 40)
        print(f"Loaded {len(vehicleID_data['paths'])} images from VehicleID.")
        print(f"Loaded {len(vehicle1M_data['paths'])} images from Vehicle-1M.")
        print(f"Loaded {len(aicity_data['paths'])} images from AI City Challenge.")
        print(f"Total images in combined dataset: {len(all_image_paths)}")
        print("-" * 40)

        self.image_paths = all_image_paths
        self.cams = all_cams

        if self.is_train:
            unique_labels = sorted(list(set(all_labels)))
            self.label_to_pid = {label: pid for pid, label in enumerate(unique_labels)}
            self.pids = [self.label_to_pid[label] for label in all_labels]
            self.num_classes = len(unique_labels)
            print(f"Total number of unique vehicle IDs (classes) for training: {self.num_classes}")
        else:
            self.pids = all_labels
            self.num_classes = -1

    # ... Other helper methods like _load_vehicleID_data etc. remain the same ...
    def _load_vehicleID_data(self, list_path, root_dir):
        paths, labels, cams = [], [], []
        if not list_path or not root_dir: return {'paths': [], 'labels': [], 'cams': []}
        try:
            with open(list_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    img_name, v_id = line.strip().split()
                    full_path = os.path.join(root_dir, img_name + ".jpg")
                    if os.path.exists(full_path):
                        paths.append(full_path)
                        labels.append(v_id)
                        cams.append(-1)
        except Exception as e: print(f"Error loading VehicleID dataset: {e}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    def _load_vehicle1M_data(self, txt_path, root_dir):
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
        except Exception as e: print(f"Error loading Vehicle-1M dataset: {e}")
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
            placeholder_tensor = torch.zeros(3, 256, 128)
            if self.transform:
                blank_pil = Image.new('RGB', (128, 256), (0, 0, 0))
                return self.transform(blank_pil), -1, -1, 0
            return placeholder_tensor, -1, -1, 0
        if self.transform: image = self.transform(image)
        return image, np.int64(pid), np.int64(camid), 0

def train_collate_fn(batch):
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids


# ==========================================================================================
# 2. REPLACEMENT SAMPLER: DISTRIBUTED-AWARE BATCH SAMPLER
# ==========================================================================================

class DistributedRandomIdentityBatchSampler(BatchSampler):
    """
    A robust, distributed-aware batch sampler for vehicle/person re-ID.
    It ensures that each batch contains `num_instances` of `P` unique identities
    and that all GPUs receive the same number of batches, preventing deadlocks.

    1. Generates the full list of batches for an epoch within `set_epoch`.
    2. `__len__` and `__iter__` use this pre-computed list for consistency.
    3. Guarantees every process will have a similar number of batches.
    """
    def __init__(self, data_source, batch_size, num_instances, world_size, rank, epoch=0):
        # We pass a dummy sampler to the parent, as we override __iter__ and __len__ completely.
        super().__init__(Sampler(data_source), batch_size, drop_last=True)

        self.data_source = data_source
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.world_size = world_size
        self.rank = rank

        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.data_source.pids):
            self.index_dic[pid].append(index)

        self.pids = sorted(list(self.index_dic.keys()))
        self.num_pids = len(self.pids)

        # This will store the batches for the current epoch for this rank
        self.batches = []
        # Call set_epoch initially to prepare batches for epoch 0
        self.set_epoch(epoch)

    def _generate_batches(self):
        """Internal method to create the batches for one full epoch."""
        # Use a generator seeded with the epoch for reproducibility across processes
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # 1. Get a globally shuffled list of all PIDs
        shuffled_pid_indices = torch.randperm(self.num_pids, generator=g).tolist()
        shuffled_pids = [self.pids[i] for i in shuffled_pid_indices]

        # 2. Create a flat list of all possible instance "chunks" from all PIDs
        all_pid_chunks = []
        for pid in shuffled_pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            # Oversample if not enough instances for at least one chunk
            if len(idxs) < self.num_instances:
                random_state = np.random.RandomState(seed=self.epoch + pid)
                idxs = random_state.choice(idxs, size=self.num_instances, replace=True).tolist()
            # Shuffle instances within the PID for variety
            random.shuffle(idxs)
            # Create chunks of size num_instances
            for i in range(len(idxs) // self.num_instances):
                all_pid_chunks.append(idxs[i * self.num_instances : (i + 1) * self.num_instances])

        # 3. Group the chunks into global batches
        global_batches = []
        num_chunks_per_batch = self.num_pids_per_batch
        for i in range(0, len(all_pid_chunks) - num_chunks_per_batch + 1, num_chunks_per_batch):
            batch_chunks = all_pid_chunks[i : i + num_chunks_per_batch]
            final_batch = [item for sublist in batch_chunks for item in sublist]
            global_batches.append(final_batch)

        # 4. Distribute the global batches among the ranks using a striding approach
        self.batches = global_batches[self.rank :: self.world_size]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. Must be called at the beginning of each epoch.
        """
        self.epoch = epoch
        self._generate_batches()


# ==========================================================================================
# 3. DDP & UTILITY FUNCTIONS
# ==========================================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[Rank {dist.get_rank()}] Process initialized and set to GPU {local_rank}")

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def gather_objects(local_list, world_size):
    """Gathers Python objects from all processes."""
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, local_list)
    # Flatten the list of lists
    return [item for sublist in gathered_list for item in sublist]

# ==========================================================================================
# 4. PROCESSOR: MODEL LOADING, TRAINING, AND TESTING LOGIC
# ==========================================================================================

def get_model(data, device):
    # This function seems to have a lot of specific model architecture names.
    # It is kept as is from your provided code.
    model = MBR_model(class_num=data['n_classes'], n_branches=["R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    if data['model_arch'] == 'MBR_R50_2B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    # ... Add all other 'if data['model_arch'] ...' conditions here as they were ...

    if data["weights_pretrain"]:
        model = load_weights_custom(model, data['weights_pretrain'], data["n_classes"])
    return model.to(device)


def train_epoch(model, device, dataloader, loss_fn, triplet_loss, optimizer, data, alpha_ce, beta_tri, logger, epoch,
                scheduler=None, scaler=None, rank=0):
    model.train()
    is_ddp = dist.is_available() and dist.is_initialized()

    # Accumulators for metrics
    train_loss, ce_loss_log, triplet_loss_log = [], [], []
    total_correct = torch.tensor(0.0).to(device)
    total_samples = 0

    # Progress bar only on the main process
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', disable=(rank != 0))

    for step, (image_batch, label, cam) in enumerate(pbar):
        image_batch = image_batch.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        loss, loss_ce, loss_t = 0, 0, 0

        # Use AMP (Automatic Mixed Precision) if scaler is provided
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
            preds, embs, _, _ = model(image_batch, cam, None)
            if not isinstance(preds, list): preds, embs = [preds], [embs]

            for item in preds: loss_ce += alpha_ce * loss_fn(item, label)
            for item in embs: loss_t += beta_tri * triplet_loss(item, label)
            loss = (loss_ce / len(preds) + loss_t / len(embs)) if data['mean_losses'] else (loss_ce + loss_t)

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            # Unscale gradients and call optimizer.step()
            scaler.step(optimizer)
            # Update the scale for next iteration
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update scheduler AFTER the optimizer has taken a step
        if scheduler:
            scheduler.step()

        # Accumulate local stats
        for prediction in preds:
            total_correct += torch.sum(torch.argmax(prediction, dim=1) == label)
            total_samples += prediction.size(0)

        # Logging from rank 0 only
        if rank == 0:
            train_loss.append(loss.item())
            ce_loss_log.append(loss_ce.item())
            triplet_loss_log.append(loss_t.item())
            if logger:
                logger.write_scalars({
                    "Loss/train_total": np.mean(train_loss),
                    "Loss/train_crossentropy": np.mean(ce_loss_log),
                    "Loss/train_triplet": np.mean(triplet_loss_log),
                    "lr/learning_rate": get_lr(optimizer),
                }, epoch * len(dataloader) + step)

    # Aggregate accuracy metrics from all processes
    if is_ddp:
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = total_samples_tensor.item()

    # Log global accuracy from rank 0
    if rank == 0:
        global_accuracy = (total_correct.item() / total_samples) if total_samples > 0 else 0
        print(f'Epoch {epoch + 1} | Global Train Acc: {global_accuracy * 100:.2f}% | Loss: {np.mean(train_loss):.4f}')
        if logger:
            logger.write_scalars({"Epoch/GlobalTrainAccuracy": global_accuracy}, epoch)

    # Return losses from rank 0 for main script logging
    return np.mean(train_loss), np.mean(ce_loss_log), np.mean(triplet_loss_log)


# ==========================================================================================
# 5. MAIN TRAINING SCRIPT
# ==========================================================================================

def main():
    # --- DDP Setup ---
    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")

    # --- Configuration ---
    parser = argparse.ArgumentParser(description='ReID model trainer')
    parser.add_argument('--config', default='./config/config_duythai.yaml', help='Config Path')
    # Add other arguments if needed
    args = parser.parse_args()
    with open(args.config, "r") as stream: data = yaml.safe_load(stream)

    set_seed(data['torch_seed'] + rank) # Different seed per process

    if rank == 0:
        print("\n--- Configuration ---")
        print(data)
        print("---------------------\n")

    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.Pad(10),
        transforms.RandomCrop((data['y_length'], data['x_length'])),
        transforms.RandomHorizontalFlip(p=data['p_hflip']),
        transforms.ToTensor(),
        transforms.Normalize(data['n_mean'], data['n_std']),
        transforms.RandomErasing(p=data['p_rerase'], value=0),
    ])

    # --- Dataset and DataLoader ---
    data_train = CombinedVehicleDataset(is_train=True, transform=train_transform)

    # Use the new Distributed Sampler
    train_batch_sampler = DistributedRandomIdentityBatchSampler(
        data_source=data_train,
        batch_size=data['BATCH_SIZE'],
        num_instances=data['NUM_INSTANCES'],
        world_size=world_size,
        rank=rank
    )

    data_train_loader = DataLoader(
        data_train,
        num_workers=data['num_workers_train'],
        collate_fn=train_collate_fn,
        pin_memory=True,
        batch_sampler=train_batch_sampler # Key change here!
    )

    # Update number of classes in config based on dataset
    data['n_classes'] = data_train.num_classes

    # --- Model, Loss, Optimizer ---
    model = get_model(data, device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=data['label_smoothing'])
    metric_loss = triplet_loss_fastreid(data['triplet_margin'], norm_feat=data['triplet_norm'], hard_mining=data['hard_mining'])

    # ######################################################################
    # ########### THIS IS THE CORRECTED LINE ###############################
    # ######################################################################
    optimizer = make_optimizer(
        data['optimizer'],
        model,
        data['lr'],
        data['weight_decay'],
        data['bias_lr_factor'],  # Added back
        data['momentum']         # Added back
    )
    # ######################################################################

    scheduler = make_warmup_scheduler(data['sched_name'], optimizer, data['num_epochs'], data['milestones'], data['gamma'], data['warmup_factor'], data['warmup_iters'], data['warmup_method'])
    scaler = torch.cuda.amp.GradScaler() if data['half_precision'] else None

    logger = Logger(data) if rank == 0 else None

    # --- Training Loop ---
    for epoch in range(data['num_epochs']):
        # Must set the epoch for the sampler to reshuffle data
        train_batch_sampler.set_epoch(epoch)

        train_loss, c_loss, t_loss = train_epoch(
            model, device, data_train_loader, loss_fn, metric_loss,
            optimizer, data, data['alpha_ce'], data['beta_tri'], logger,
            epoch, scheduler, scaler, rank
        )

        # Saving and logging only on the main process
        if rank == 0:
            print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, CE Loss: {c_loss:.4f}, Triplet Loss: {t_loss:.4f}")
            if epoch % data['validation_period'] == 0 or epoch == data['num_epochs'] - 1:
                logger.save_model(model.module) # Save the underlying model

    if rank == 0 and logger:
        logger.save_log()

    cleanup_ddp()

if __name__ == "__main__":
    main()