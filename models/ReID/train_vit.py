# ==========================================================================================
# train_vit_classifier.py
# A standalone script to train a ViT as a powerful classifier for Re-ID using a combined dataset.
#
# How to Run (using 2 GPUs):
# 1. Save this file as `train_vit_classifier.py`.
# 2. Save the new YAML config as `config_vit_combined.yaml`.
# 3. Adjust data paths in the config file.
# 4. Install dependencies: pip install torch torchvision pyyaml opencv-python
# 5. Run from your terminal:
#    torchrun --nproc_per_node=2 train_vit_classifier.py --config ./config_vit_combined.yaml
# ==========================================================================================

import os
import random
import argparse
import yaml
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Dict, List

# New imports required by the dataset class
import cv2
from pathlib import Path
import collections

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

import torch.multiprocessing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.multiprocessing.set_sharing_strategy('file_system')


# ==========================================================================================
# SECTION 1: MODEL DEFINITION (ViTForReID)
# ==========================================================================================
class ViTForReID(nn.Module):
    def __init__(self, num_classes: int):
        super(ViTForReID, self).__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.backbone = vit_b_16(weights=weights)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor, cam=None, view=None):
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        return [logits], [embedding]  # Return logits and embeddings


# ==========================================================================================
# SECTION 2: COMBINED DATASET CLASS (Your new class)
# ==========================================================================================
class CombinedVehicleDataset(Dataset):
    def __init__(self,
                 vehicleID_list_path, vehicleID_root_dir,
                 vehicle1M_txt_path, vehicle1M_root_dir,
                 aicity_root_path, aicity_crop_dir, aicity_split,
                 is_train=True, transform=None):

        self.transform = transform
        self.is_train = is_train

        vehicleID_data = self._load_vehicleID_data(vehicleID_list_path, vehicleID_root_dir)
        vehicle1M_data = self._load_vehicle1M_data(vehicle1M_txt_path, vehicle1M_root_dir)
        aicity_data = self._load_aicity_data(aicity_root_path, aicity_crop_dir, aicity_split)

        all_image_paths = (vehicleID_data['paths'] + vehicle1M_data['paths'] + aicity_data['paths'])
        all_labels = (
                [f"vehicleid_{l}" for l in vehicleID_data['labels']] +
                [f"vehicle1m_{l}" for l in vehicle1M_data['labels']] +
                [f"aicity_{l}" for l in aicity_data['labels']]
        )
        all_cams = (vehicleID_data['cams'] + vehicle1M_data['cams'] + aicity_data['cams'])

        print("-" * 40)
        print(f"Loaded {len(vehicleID_data['paths'])} images from VehicleID.")
        print(f"Loaded {len(vehicle1M_data['paths'])} images from Vehicle-1M.")
        print(f"Loaded {len(aicity_data['paths'])} images from AICity22.")
        print(f"Total images in combined dataset: {len(all_image_paths)}")
        print("-" * 40)

        self.image_paths = all_image_paths
        self.cams = all_cams

        unique_labels = sorted(list(set(all_labels)))
        self.label_to_pid = {label: pid for pid, label in enumerate(unique_labels)}
        self.pids = [self.label_to_pid[label] for label in all_labels]
        self.num_classes = len(unique_labels)
        print(f"Total number of unique vehicle IDs (classes) for training: {self.num_classes}")

    def _load_vehicleID_data(self, list_path, root_dir):
        paths, labels, cams = [], [], []
        if not list_path or not os.path.exists(list_path): return {'paths': [], 'labels': [], 'cams': []}
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
        except Exception as e:
            print(f"Error loading VehicleID: {e}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    def _load_vehicle1M_data(self, txt_path, root_dir):
        paths, labels, cams = [], [], []
        if not txt_path or not os.path.exists(txt_path): return {'paths': [], 'labels': [], 'cams': []}
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
            print(f"Error loading Vehicle-1M: {e}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    def _load_aicity_data(self, root_dir: str, crop_dir: str, split: str) -> Dict:
        root_path = Path(root_dir)
        crop_path = Path(crop_dir)
        paths, labels, cams = [], [], []

        if not root_path.exists(): return {'paths': [], 'labels': [], 'cams': []}

        if crop_path.exists() and any(crop_path.iterdir()):
            print(f"INFO: Loading cached AICity crops from: {crop_path}")
            for img_file in sorted(list(crop_path.glob("*.jpg"))):
                try:
                    parts = img_file.stem.split('_')
                    cam_id_str, vehicle_id_str = parts[0], parts[1]
                    paths.append(str(img_file))
                    labels.append(vehicle_id_str.replace('v', ''))
                    cams.append(int(cam_id_str.replace('c', '')))
                except (ValueError, IndexError):
                    continue
            if paths: return {'paths': paths, 'labels': labels, 'cams': cams}

        print(f"WARNING: AICity cache dir '{crop_path}' empty. Starting one-time crop extraction.")
        print("         This can take several hours depending on your system.")
        crop_path.mkdir(parents=True, exist_ok=True)

        vehicle_data = self._group_aicity_vehicle_ids(root_path, split=split)
        video_caps = {}
        try:
            for vehicle_id, sightings in tqdm(vehicle_data.items(), desc="Extracting AICity Crops"):
                for sighting in sightings:
                    video_file, frame_num, bbox = sighting['video_path'], sighting['frame_num'], sighting['bbox']
                    try:
                        cam_id = int(Path(video_file).parent.name.replace('c', ''))
                    except (ValueError, IndexError):
                        continue

                    if video_file not in video_caps:
                        cap = cv2.VideoCapture(video_file)
                        if not cap.isOpened(): video_caps[video_file] = None; continue
                        video_caps[video_file] = cap

                    cap = video_caps.get(video_file)
                    if cap is None: continue

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
        finally:
            for cap in video_caps.values():
                if cap: cap.release()
        return {'paths': paths, 'labels': labels, 'cams': cams}

    @staticmethod
    def _group_aicity_vehicle_ids(dataset_path: Path, split: str) -> Dict[int, List[Dict]]:
        vehicle_data = collections.defaultdict(list)
        split_path = dataset_path / split
        if not split_path.exists(): return {}

        for cam_path in split_path.iterdir():
            if not cam_path.is_dir(): continue
            gt_path = cam_path / 'gt/gt.txt'
            video_path = cam_path / 'vdo.avi'
            if not gt_path.exists() or not video_path.exists(): continue

            with open(gt_path, 'r') as f:
                for line in f:
                    try:
                        frame, v_id, x, y, w, h = map(int, line.strip().split(',')[:6])
                        if v_id == -1: continue
                        vehicle_data[v_id].append(
                            {'video_path': str(video_path), 'frame_num': frame, 'bbox': [x, y, w, h]})
                    except ValueError:
                        continue
        return dict(vehicle_data)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        pid = self.pids[idx]
        camid = self.cams[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, np.int64(pid), np.int64(camid)


# --- UPDATED COLLATE FUNCTION ---
def train_collate_fn(batch):
    # The dataset __getitem__ returns (image, pid, camid). We only need image and pid.
    imgs, pids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


# ==========================================================================================
# SECTION 3: DDP HELPERS AND UTILITIES
# ==========================================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def cleanup_ddp():
    dist.destroy_process_group()


# ==========================================================================================
# SECTION 4: TRAINING LOGIC
# ==========================================================================================
def train_epoch(model, device, dataloader, loss_fn, optimizer, scheduler, scaler, epoch, rank):
    model.train()
    total_loss = 0.0
    total_correct = torch.tensor(0.0).to(device)
    total_samples = 0
    is_ddp = dist.is_available() and dist.is_initialized()

    if is_ddp:
        dataloader.sampler.set_epoch(epoch)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', disable=(rank != 0))

    for image_batch, label in pbar:
        image_batch = image_batch.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
            preds, _ = model(image_batch)
            loss = loss_fn(preds[0], label)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        total_correct += torch.sum(torch.argmax(preds[0], dim=1) == label)
        total_samples += image_batch.size(0)

    if is_ddp:
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = total_samples_tensor.item()

    if rank == 0:
        avg_loss = total_loss / len(dataloader)
        accuracy = (total_correct.item() / total_samples) * 100
        print(f"Epoch {epoch + 1} | Global Train Acc: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}")


# ==========================================================================================
# SECTION 5: MAIN EXECUTION SCRIPT
# ==========================================================================================
def main():
    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")

    parser = argparse.ArgumentParser(description='ViT ReID Classifier Trainer')
    parser.add_argument('--config', default='./config_vit_combined.yaml', help='Config Path')
    args = parser.parse_args()
    with open(args.config, "r") as stream:
        data = yaml.safe_load(stream)

    set_seed(data['torch_seed'] + rank)

    if rank == 0:
        print("\n--- Configuration ---\n", data, "\n---------------------\n")

    train_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.RandomHorizontalFlip(p=data['p_hflip']),
        transforms.Pad(10),
        transforms.RandomCrop((data['y_length'], data['x_length'])),
        transforms.ToTensor(),
        transforms.Normalize(data['n_mean'], data['n_std']),
        transforms.RandomErasing(p=data['p_rerase']),
    ])

    # --- UPDATED DATASET INITIALIZATION ---
    data_train = CombinedVehicleDataset(
        transform=train_transform,
        is_train=True,
        vehicleID_list_path=data.get('vehicleID_list_path'),
        vehicleID_root_dir=data.get('vehicleID_root_dir'),
        vehicle1M_txt_path=data.get('vehicle1M_txt_path'),
        vehicle1M_root_dir=data.get('vehicle1M_root_dir'),
        aicity_root_path=data.get('aicity_root_path'),
        aicity_crop_dir=data.get('aicity_crop_dir'),
        aicity_split=data.get('aicity_split', 'train')
    )

    train_sampler = DistributedSampler(data_train, num_replicas=world_size, rank=rank, shuffle=True)
    data_train_loader = DataLoader(
        data_train, batch_size=data['BATCH_SIZE'], num_workers=data['num_workers_train'],
        sampler=train_sampler, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
    )

    model = ViTForReID(num_classes=data_train.num_classes).to(device)
    state_dict = torch.load("vit_classifier_combined_epoch_20.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = DDP(model, device_ids=[local_rank])

    loss_fn = nn.CrossEntropyLoss(label_smoothing=data['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=data['lr'], weight_decay=data['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=data['milestones'], gamma=data['gamma'])
    scaler = torch.cuda.amp.GradScaler(enabled=data['half_precision'])

    for epoch in range(data['num_epochs']):
        train_epoch(model, device, data_train_loader, loss_fn, optimizer, scheduler, scaler, epoch, rank)

        if rank == 0 and (epoch + 1) % data.get('save_period', 5) == 0:
            save_path = f'vit_classifier_combined_epoch_{epoch + 1}.pt'
            torch.save(model.module.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()