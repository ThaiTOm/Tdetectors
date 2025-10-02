import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# ### DDP CHANGE ###: Import the correct sampler and DDP modules
from dataset import CombinedVehicleDataset, train_collate_fn, DistributedRandomIdentityBatchSampler
from loss.losses import triplet_loss_fastreid
from lr_scheduler.sche_optim import make_optimizer, make_warmup_scheduler
import argparse
import torch.multiprocessing
import yaml
import os
from tensorboard_log import Logger
from processor import get_model, train_epoch, test_epoch
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.multiprocessing.set_sharing_strategy('file_system')


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ### DDP CHANGE ###: Add setup and cleanup functions for the distributed environment
def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[Rank {dist.get_rank()}] Process initialized and set to GPU {local_rank}")


def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


if __name__ == "__main__":
    # ### DDP CHANGE ###: Initialize the distributed environment first
    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    parser = argparse.ArgumentParser(description='ReID model trainer')
    # ... (All your argument parsing code remains exactly the same) ...
    parser.add_argument('--config', default=None, help='Config Path')
    parser.add_argument('--batch_size', default=None, type=int, help='Batch size')
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
    parser.add_argument('--parallel', default=None, help='Whether to used DataParallel for multi-gpu in one device')
    parser.add_argument('--half_precision', default=None, help='Use of mixed precision')
    parser.add_argument('--mean_losses', default=None, help='Use of mixed precision')

    args = parser.parse_args()

    ### Load hyper parameters
    if args.config:
        with open(args.config, "r") as stream:
            data = yaml.safe_load(stream)
    else:
        with open("./config/config_duythai.yaml", "r") as stream:
            data = yaml.safe_load(stream)

    # ... (All your config overriding code remains exactly the same) ...
    data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    data['p_hflip'] = args.hflip or data['p_hflip']
    data['y_length'] = args.imgsize_y or data['y_length']
    data['x_length'] = args.imgsize_x or data['x_length']
    data['p_rerase'] = args.randomerase or data['p_rerase']
    data['dataset'] = args.dataset or data['dataset']
    data['NUM_INSTANCES'] = args.num_instances or data['NUM_INSTANCES']
    data['model_arch'] = args.model_arch or data['model_arch']
    if args.triplet_margin is not None: data['triplet_margin'] = data['triplet_margin']
    data['softmax_loss'] = args.softmax_loss or data['softmax_loss']
    data['metric_loss'] = args.metric_loss or data['metric_loss']
    data['optimizer'] = args.optimizer or data['optimizer']
    data['lr'] = args.initial_lr or data['lr']
    data['parallel'] = args.parallel or data['parallel']
    data['alpha_ce'] = args.lambda_ce or data['alpha_ce']
    data['beta_tri'] = args.lambda_triplet or data['beta_tri']
    data['backbone'] = args.backbone or data['backbone']
    data['half_precision'] = args.half_precision or data['half_precision']
    if args.mean_losses is not None: data['mean_losses'] = bool(args.mean_losses)

    alpha_ce = data['alpha_ce']
    beta_tri = data['beta_tri']

    #### Set Seed for consistent and deterministic results
    # ### DDP CHANGE ###: Add rank to seed to ensure different processes have different random states
    set_seed(data['torch_seed'] + rank)

    ### Config print
    if rank == 0:
        print("\n\n\n  Config used: \n")
        print(data)
        print("\n\n\n End config")

    #### Transformation augmentation (remains the same)
    teste_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(data['n_mean'], data['n_std']),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.Pad(10),
        transforms.RandomCrop((data['y_length'], data['x_length'])),
        transforms.RandomHorizontalFlip(p=data['p_hflip']),
        transforms.ToTensor(),
        transforms.Normalize(data['n_mean'], data['n_std']),
        transforms.RandomErasing(p=data['p_rerase'], value=0),
    ])

    #### Dataset Loading
    data_train = CombinedVehicleDataset(is_train=True, transform=train_transform)
    if rank == 0:
        print("Total dataset length:", len(data_train))

    # ### DDP CHANGE ###: Use the DistributedRandomIdentityBatchSampler
    train_batch_sampler = DistributedRandomIdentityBatchSampler(
        data_source=data_train,
        batch_size=data['BATCH_SIZE'],
        num_instances=data['NUM_INSTANCES'],
        world_size=world_size,
        rank=rank,
        epoch=0  # Initial epoch
    )

    # ### DDP CHANGE ###: Create DataLoader using the `batch_sampler`
    # When using `batch_sampler`, you must NOT specify `batch_size`, `shuffle`, `sampler`, or `drop_last`.
    data_train_loader = DataLoader(
        data_train,
        num_workers=data['num_workers_train'],
        collate_fn=train_collate_fn,
        pin_memory=True,
        batch_sampler=train_batch_sampler
    )

    # ### DDP CHANGE ###: Set the device for the current process
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(f'Training on {world_size} GPUs. Main process (rank 0) is on {device}')

    # Create Model on the correct device
    model = get_model(data, device)

    # ### DDP CHANGE ###: Replace nn.DataParallel with DDP
    # The old DataParallel code is removed.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ### Losses ### (remains the same)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=data['label_smoothing'])
    metric_loss = triplet_loss_fastreid(data['triplet_margin'], norm_feat=data['triplet_norm'],
                                        hard_mining=data['hard_mining'])

    #### Optimizer (remains the same)
    optimizer = make_optimizer(data['optimizer'], model, data['lr'], data['weight_decay'], data['bias_lr_factor'],
                               data['momentum'])

    ### Schedule for the optimizer (remains the same)
    scheduler = make_warmup_scheduler(data['sched_name'], optimizer, data['num_epochs'], data['milestones'],
                                      data['gamma'],
                                      data['warmup_factor'], data['warmup_iters'], data['warmup_method'], last_epoch=-1,
                                      min_lr=data['min_lr']) if data['epoch_freeze_L1toL3'] == 0 else None

    ### If running with fp16 precision
    scaler = torch.amp.GradScaler('cuda') if data['half_precision'] else None

    ### Initiate a Logger (only on the main process)
    logger = Logger(data) if rank == 0 else None

    ## Freezing logic (remains the same, but now operates on `model.module`)
    # if data['freeze_backbone_warmup']:
    #     for param in model.module.modelup2L3.parameters(): param.requires_grad = False
    #     for param in model.module.modelL4.parameters(): param.requires_grad = False
    # if data['epoch_freeze_L1toL3'] > 0:
    #     for param in model.module.modelup2L3.parameters(): param.requires_grad = False
    #     if rank == 0: print("\nFroze Backbone before branches!")

    ## Training Loop
    # ### DDP CHANGE ###: Disable tqdm on non-main processes
    for epoch in tqdm(range(data['num_epochs']), disable=(rank != 0)):
        # ### DDP CHANGE ###: Set the epoch for the sampler to ensure shuffling
        train_batch_sampler.set_epoch(epoch)

        # Unfreezing logic (remains the same, operates on `model.module`)
        # if epoch == data['warmup_iters'] - 1:
        #     for param in model.module.modelup2L3.parameters(): param.requires_grad = True
        #     for param in model.module.modelL4.parameters(): param.requires_grad = True
        #
        # if epoch == data['epoch_freeze_L1toL3'] - 1:
        #     # ... scheduler recreation ...
        #     for param in model.module.modelup2L3.parameters(): param.requires_grad = True
        #     if rank == 0: print("\nUnfrozen Backbone before branches!")

        ### Train Loop
        # ### DDP CHANGE ###: Pass the rank to the train_epoch function for logging
        train_loss, c_loss, t_loss, _, _ = train_epoch(
            model, device, data_train_loader, loss_fn, metric_loss,
            optimizer, data, alpha_ce, beta_tri, logger,
            epoch, scheduler, scaler, rank=rank
        )

        ### CORRECTED LOGIC: Step the scheduler AFTER the training epoch
        if scheduler is not None:
            if epoch >= data['epoch_freeze_L1toL3'] - 1:
                scheduler.step()

        ### Evaluation and Saving (only on the main process)
        if rank == 0:
            # if epoch % data['validation_period'] == 0 or epoch >= data['num_epochs'] - 15:
            #     print('\n EPOCH {}/{} \t train loss {} \t Classification loss {} \t Triplet loss {}'.format(
            #         epoch + 1, data['num_epochs'], train_loss, c_loss, t_loss,
            #     ))
                # ### DDP CHANGE ###: Save the underlying model's state dict
            logger.save_model(model.module)

    if rank == 0:
        print("Best mAP: ", np.max(logger.logscalars['Accuraccy/mAP']))
        print("Best CMC1: ", np.max(logger.logscalars['Accuraccy/CMC1']))
        logger.save_log()

    # ### DDP CHANGE ###: Clean up the distributed environment
    cleanup_ddp()