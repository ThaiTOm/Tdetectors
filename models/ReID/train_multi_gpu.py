import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset  ### NEW/MODIFIED ###
from tqdm import tqdm
import numpy as np
from dataset import CombinedVehicleDataset, train_collate_fn, RandomIdentitySampler, \
    DistributedRandomIdentityBatchSampler
from loss.losses import triplet_loss_fastreid
from lr_scheduler.sche_optim import make_optimizer, make_warmup_scheduler
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import yaml
import os
from tensorboard_log import Logger
from processor import get_model, train_epoch, test_epoch
import random
import copy  ### NEW/MODIFIED ###


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, data):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    if rank == 0:
        print("\n\n\n  Config used: \n")
        print(data)
        print("\n\n\n End config")

    set_seed(data['torch_seed'])

    ### NEW/MODIFIED ###: Define separate transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.RandomHorizontalFlip(p=data['p_hflip']),
        transforms.Pad(10),
        transforms.RandomCrop((data['y_length'], data['x_length'])),  # Add random crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ### NEW/MODIFIED ###: Create full dataset and then split it
    # Step 1: Create two instances of the full dataset. One for training, one for validation.
    # We use deepcopy to ensure they are independent objects.
    full_dataset_train_version = CombinedVehicleDataset(is_train=True, transform=train_transform)
    full_dataset_val_version = copy.deepcopy(full_dataset_train_version)
    full_dataset_val_version.transform = val_transform  # Assign the validation transform

    # Step 2: Perform an identity-aware split on the PIDs
    unique_pids = sorted(list(set(full_dataset_train_version.pids)))
    random.shuffle(unique_pids)

    val_split_ratio = data.get('validation_split_ratio', 0.1)
    split_point = int(len(unique_pids) * (1 - val_split_ratio))

    train_pids = set(unique_pids[:split_point])
    val_pids = set(unique_pids[split_point:])

    if rank == 0:
        print(f"Splitting dataset: {len(train_pids)} IDs for training, {len(val_pids)} IDs for validation.")

    # Step 3: Create lists of indices for the train and validation subsets
    train_indices = [i for i, pid in enumerate(full_dataset_train_version.pids) if pid in train_pids]
    val_indices = [i for i, pid in enumerate(full_dataset_train_version.pids) if pid in val_pids]

    # Step 4: Create PyTorch Subset objects
    dataset_train = Subset(full_dataset_train_version, train_indices)
    dataset_val = Subset(full_dataset_val_version, val_indices)

    # We need to adapt the custom sampler to work with the Subset
    # A simple way is to give the sampler the necessary attributes from the subset.
    dataset_train.pids = [full_dataset_train_version.pids[i] for i in train_indices]

    # Create the training sampler using the new subset
    train_sampler = DistributedRandomIdentityBatchSampler(
        data_source=dataset_train,
        batch_size=data['BATCH_SIZE'],
        num_instances=data['NUM_INSTANCES'],
        world_size=world_size,
        rank=rank,
        epoch=0
    )

    data_train_loader = DataLoader(
        dataset_train,
        batch_sampler=train_sampler,
        num_workers=data['num_workers_train'],
        collate_fn=train_collate_fn,
        drop_last=True

    )

    ### NEW/MODIFIED ###: Create validation loader (only needed for rank 0)
    data_val_loader = None
    if rank == 0:
        # No special sampler needed for validation, just a simple sequential pass.
        # batch_size for validation can often be larger.
        val_batch_size = data.get('val_batch_size', data['BATCH_SIZE'] * 2)
        data_val_loader = DataLoader(
            dataset_val,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=data['num_workers_train']
        )

    device = torch.device(f"cuda:{rank}")
    print(f'Rank {rank} assigned to device: {device}')

    model = get_model(data, device)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=data['label_smoothing'])
    metric_loss = triplet_loss_fastreid(data['triplet_margin'], norm_feat=data['triplet_norm'],
                                        hard_mining=data['hard_mining'])

    optimizer = make_optimizer(data['optimizer'], model, data['lr'], data['weight_decay'], data['bias_lr_factor'],
                               data['momentum'])

    scheduler = None
    if data['epoch_freeze_L1toL3'] == 0:
        scheduler = make_warmup_scheduler(data['sched_name'], optimizer, data['num_epochs'], data['milestones'],
                                          data['gamma'], data['warmup_factor'], data['warmup_iters'],
                                          data['warmup_method'], last_epoch=-1, min_lr=data['min_lr'])

    scaler = torch.cuda.amp.GradScaler() if data['half_precision'] else None
    logger = Logger(data) if rank == 0 else None

    if data['freeze_backbone_warmup']:
        for param in model.module.modelup2L3.parameters(): param.requires_grad = False
        for param in model.module.modelL4.parameters(): param.requires_grad = False
    if data['epoch_freeze_L1toL3'] > 0:
        for param in model.module.modelup2L3.parameters(): param.requires_grad = False
        if rank == 0: print("\nFroze Backbone before branches!")

    alpha_ce = data['alpha_ce']
    beta_tri = data['beta_tri']

    for epoch in range(data['num_epochs']):
        data_train_loader.batch_sampler.set_epoch(epoch)

        if epoch == data['warmup_iters'] - 1:
            for param in model.module.modelup2L3.parameters(): param.requires_grad = True
            for param in model.module.modelL4.parameters(): param.requires_grad = True

        if epoch == data['epoch_freeze_L1toL3'] - 1:
            scheduler = make_warmup_scheduler(data['sched_name'], optimizer, data['num_epochs'], data['milestones'],
                                              data['gamma'], data['warmup_factor'], data['warmup_iters'],
                                              data['warmup_method'], last_epoch=-1, min_lr=data['min_lr'])
            for param in model.module.modelup2L3.parameters(): param.requires_grad = True
            if rank == 0: print("\nUnfrozen Backbone before branches!")

        if scheduler and epoch >= data['epoch_freeze_L1toL3'] - 1:
            scheduler.step()

        train_loss, c_loss, t_loss, alpha_ce, beta_tri = train_epoch(model, device, data_train_loader, loss_fn,
                                                                     metric_loss, optimizer, data, alpha_ce, beta_tri,
                                                                     logger, epoch, scheduler, scaler, rank)

        ### NEW/MODIFIED ###: Run validation on rank 0
        if rank == 0:
            print(
                f'\nEPOCH {epoch + 1}/{data["num_epochs"]} | Train Loss: {train_loss:.4f} | '
                f'Class Loss: {c_loss:.4f} | Triplet Loss: {t_loss:.4f}'
            )

            if epoch % data['validation_period'] == 0 or epoch >= data['num_epochs'] - 5:
                print("Running validation...")
                # The test_epoch function should handle model.eval() and torch.no_grad()
                # We use model.module to pass the underlying model, not the DDP wrapper
                test_epoch(model.module, device, data_val_loader, logger, epoch, data)

                # Save the model based on validation performance (e.g., if mAP is best)
                # This logic would be inside logger.save_model if it's designed that way
                logger.save_model(model.module)

            # Also save last model
            if epoch == data['num_epochs'] - 1:
                logger.save_model(model.module, last=True)

    if rank == 0:
        print("Best mAP: ", np.max(logger.logscalars.get('Accuraccy/mAP', [0])))
        print("Best CMC1: ", np.max(logger.logscalars.get('Accuraccy/CMC1', [0])))
        logger.save_log()

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReID model trainer')
    parser.add_argument('--config', default=None, help='Config Path')
    parser.add_argument('--batch_size', default=None, type=int, help='Batch size PER GPU')
    parser.add_argument('--backbone', default=None, help='Model Backbone')
    parser.add_argument('--hflip', default=None, type=float, help='Probabilty for horizontal flip')
    parser.add_argument('--randomerase', default=None, type=float, help='Probabilty for random erasing')
    parser.add_argument('--dataset', default=None, help='Choose one of [Veri776, VERIWILD, Market1501, VehicleID]')
    parser.add_argument('--imgsize_x', default=None, type=int, help='width image')
    parser.add_argument('--imgsize_y', default=None, type=int, help='height image')
    parser.add_argument('--num_instances', default=None, type=int,
                        help='Number of images belonging to an ID inside of batch, the numbers of IDs is batch_size/num_instances')
    parser.add_argument('--model_arch', default=None, help='Model Architecture')
    parser.add_argument('--softmax_loss', default=None, help='The loss used for classification')
    parser.add_argument('--metric_loss', default=None, help='The loss used as metric loss')
    parser.add_argument("--triplet_margin", default=None, type=float,
                        help='With margin>0 uses normal triplet loss. If margin<=0 or None Soft Margin Triplet Loss is used instead!')
    parser.add_argument('--optimizer', default=None, help='Adam or SGD')
    parser.add_argument('--initial_lr', default=None, type=float, help='Initial learning rate after warm-up')
    parser.add_argument('--lambda_ce', default=None, type=float, help='multiplier of the classification loss')
    parser.add_argument('--lambda_triplet', default=None, type=float, help='multiplier of the metric loss')
    parser.add_argument('--half_precision', default=None, help='Use of mixed precision')
    parser.add_argument('--mean_losses', default=None, help='Use of mixed precision')
    ### NEW/MODIFIED ###
    parser.add_argument('--validation_split_ratio', default=None, type=float,
                        help='Ratio of the dataset to be used for validation (e.g., 0.1 for 10%)')

    args = parser.parse_args()

    config_file = args.config or "./config/config_duythai.yaml"
    with open(config_file, "r") as stream:
        data = yaml.safe_load(stream)

    data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    data['p_hflip'] = args.hflip or data['p_hflip']
    data['y_length'] = args.imgsize_y or data['y_length']
    data['x_length'] = args.imgsize_x or data['x_length']
    data['p_rerase'] = args.randomerase or data['p_rerase']
    data['dataset'] = args.dataset or data['dataset']
    data['NUM_INSTANCES'] = args.num_instances or data['NUM_INSTANCES']
    data['model_arch'] = args.model_arch or data['model_arch']
    if args.triplet_margin is not None: data['triplet_margin'] = args.triplet_margin
    data['softmax_loss'] = args.softmax_loss or data['softmax_loss']
    data['metric_loss'] = args.metric_loss or data['metric_loss']
    data['optimizer'] = args.optimizer or data['optimizer']
    data['lr'] = args.initial_lr or data['lr']
    data['alpha_ce'] = args.lambda_ce or data['alpha_ce']
    data['beta_tri'] = args.lambda_triplet or data['beta_tri']
    data['backbone'] = args.backbone or data['backbone']
    data['half_precision'] = args.half_precision or data['half_precision']
    if args.mean_losses is not None: data['mean_losses'] = bool(args.mean_losses)
    ### NEW/MODIFIED ###
    data['validation_split_ratio'] = args.validation_split_ratio or data.get('validation_split_ratio', 0.1)

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Found {world_size} GPUs. Spawning DDP processes.")
        mp.spawn(main_worker,
                 args=(world_size, data),
                 nprocs=world_size,
                 join=True)
    elif world_size == 1:
        print("Found 1 GPU. Running in single-GPU mode.")
        main_worker(0, 1, data)
    else:
        print("No GPUs found. Exiting.")