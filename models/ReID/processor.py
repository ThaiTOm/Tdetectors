import torch.distributed as dist
from turtle import update
import torch
import torch.nn.functional as F
from models.models import MBR_model, load_weights_custom
from tqdm import tqdm
import numpy as np
from metrics.eval_reid import eval_func

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def get_model(data, device):

    ### 2B hybrid No LBS   
    if 'Hybrid_2B' == data['model_arch']:
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "BoT"], n_groups=0, losses="Classical", LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 2B R50 No LBS
    if 'R50_2B' == data['model_arch']:
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 2B R50 LBS
    if data['model_arch'] == 'MBR_R50_2B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### Baseline with BoT
    if data['model_arch'] == 'BoT_baseline':
        model = MBR_model(class_num=data['n_classes'], n_branches=["BoT"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 2B BoT LBS
    if data['model_arch'] == 'MBR_BOT_2B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### MBR-4B (4B hybrid LBS)
    if data['model_arch'] == 'MBR_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    
    ### 4B hybdrid No LBS
    if data['model_arch'] == 'Hybrid_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "BoT", "BoT"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4B R50 No LBS
    if data['model_arch'] == 'R50_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])    

    if data['model_arch'] == 'MBR_R50_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4G hybryd with LBS     MBR-4G
    if data['model_arch'] =='MBR_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4G hybrid No LBS
    if data['model_arch'] =='Hybrid_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] =='MBR_2x2G':    
        model = MBR_model(class_num=data['n_classes'], n_branches=['2x'], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x2g=True, group_conv_mhsa_2=True) 

    if data['model_arch'] =='MBR_R50_2x2G':  
        model = MBR_model(class_num=data['n_classes'], n_branches=['2x'], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x2g=True)  

    ### 2G BoT LBS
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], end_bot_g=True)

    ### 2G R50 LBS
    if data['model_arch'] =='MBR_R50_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 2G Hybrid No LBS
    if data['model_arch'] =='Hybrid_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], group_conv_mhsa_2=True)

    ### 2G R50 No LBS
    if data['model_arch'] =='R50_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4G R50 No LBS
    if data['model_arch'] =='R50_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4G only R50 with LBS
    if data['model_arch'] =='MBR_R50_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], group_conv_mhsa_2=True)
    
    if data['model_arch'] =='MBR_R50_2x4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=["2x"], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x4g=True)

    if data['model_arch'] =='MBR_2x4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=["2x"], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x4g=True, group_conv_mhsa=True)

    if data['model_arch'] == 'Baseline':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data["weights_pretrain"]:
        model = load_weights_custom(model, data['weights_pretrain'], data["n_classes"])

    return model.to(device)


def train_epoch(model, device, dataloader, loss_fn, triplet_loss, optimizer, data, alpha_ce, beta_tri, logger, epoch,
                scheduler=None, scaler=False, rank=0):
    model.train()
    train_loss = []
    ce_loss_log = []
    triplet_loss_log = []
    gamma_ce = data['gamma_ce']
    gamma_t = data['gamma_t']
    model_arch = data['model_arch']

    ### DDP MODIFICATION ###: Check if we are in a distributed setting
    is_ddp = dist.is_available() and dist.is_initialized()

    ### DDP MODIFICATION ###: Only the main process (rank 0) should display progress bars for logging.
    loss_log, loss_ce_log, loss_triplet_log = None, None, None
    if rank == 0:
        loss_log = tqdm(total=0, position=1, bar_format='{desc}', leave=True)
        loss_ce_log = tqdm(total=0, position=2, bar_format='{desc}', leave=True)
        loss_triplet_log = tqdm(total=0, position=3, bar_format='{desc}', leave=True)

    # These will accumulate totals for accuracy calculation
    total_correct = torch.tensor(0.0).to(device)
    total_samples = 0
    stepcount = 0

    ### DDP MODIFICATION ###: Disable the main progress bar for non-main processes.
    main_pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} (%)', bar_format='{l_bar}{bar:20}{r_bar}',
                     disable=(rank != 0))

    for image_batch, label, cam, view in main_pbar:
        image_batch = image_batch.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        loss_ce = 0
        loss_t = 0
        loss = 0

        # Forward pass logic remains the same
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds, embs, _, _ = model(image_batch, cam, view)
                if not isinstance(preds, list):
                    preds, embs = [preds], [embs]

                for i, item in enumerate(preds):
                    loss_ce += (
                                   alpha_ce if i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch else gamma_ce) * loss_fn(
                        item, label)
                for i, item in enumerate(embs):
                    loss_t += (
                                  beta_tri if i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch else gamma_t) * triplet_loss(
                        item, label)

                loss = (loss_ce / len(preds) + loss_t / len(embs)) if data['mean_losses'] else (loss_ce + loss_t)
        else:
            preds, embs, _, _ = model(image_batch, cam, view)
            if not isinstance(preds, list):
                preds, embs = [preds], [embs]

            for i, item in enumerate(preds):
                loss_ce += (
                               alpha_ce if i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch else gamma_ce) * loss_fn(
                    item, label)
            for i, item in enumerate(embs):
                loss_t += (
                              beta_tri if i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch else gamma_t) * triplet_loss(
                    item, label)

            loss = (loss_ce / len(preds) + loss_t / len(embs)) if data['mean_losses'] else (loss_ce + loss_t)

        # Accumulate accuracy stats locally
        for prediction in preds:
            total_correct += torch.sum(torch.argmax(prediction, dim=1) == label)
            total_samples += prediction.size(0)

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Logging from rank 0 only
        if rank == 0:
            loss_log.set_description_str(f'train loss : {loss.item():.3f}')
            loss_ce_log.set_description_str(f'CrossEntropy: {loss_ce.item():.3f}')
            loss_triplet_log.set_description_str(f'Triplet : {loss_t.item():.3f}')

            train_loss.append(loss.item())
            ce_loss_log.append(loss_ce.item())
            triplet_loss_log.append(loss_t.item())

            ### DDP MODIFICATION ###: Only log if the logger object exists (it won't on other ranks)
            if logger:
                # Note: this accuracy is a running accuracy on rank 0's data, which is fine for step-wise logging
                current_acc = (total_correct / total_samples).cpu().numpy()
                logger.write_scalars({
                    "Loss/train_total": np.mean(train_loss),
                    "Loss/train_crossentropy": np.mean(ce_loss_log),
                    "Loss/train_triplet": np.mean(triplet_loss_log),
                    "Loss/ce_loss_weight": alpha_ce,
                    "Loss/triplet_loss_weight": beta_tri,
                    "lr/learning_rate": get_lr(optimizer),
                    "Loss/AccuracyTrain": current_acc
                }, epoch * len(dataloader) + stepcount)
        stepcount += 1

    ### DDP MODIFICATION ###: Aggregate accuracy metrics from all processes
    if is_ddp:
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        # We also need to gather the total number of samples
        total_samples_tensor = torch.tensor(total_samples).to(device)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = total_samples_tensor.item()

    ### DDP MODIFICATION ###: Only print final accuracy on the main process
    if rank == 0:
        global_accuracy = (total_correct / total_samples).item() if total_samples > 0 else 0
        print(f'\nGlobal Train ACC (%): {global_accuracy * 100:.2f}\n')
        # This logger write is now at the end of the epoch, logging the global accuracy
        if logger:
            logger.write_scalars({"Epoch/GlobalTrainAccuracy": global_accuracy}, epoch)

    # All processes must return, but the main script will only use the values from rank 0.
    mean_train_loss = np.mean(train_loss) if rank == 0 else 0
    mean_ce_loss = np.mean(ce_loss_log) if rank == 0 else 0
    mean_triplet_loss = np.mean(triplet_loss_log) if rank == 0 else 0

    return mean_train_loss, mean_ce_loss, mean_triplet_loss, alpha_ce, beta_tri

def gather_objects(local_list, world_size):
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, local_list)
    # Flatten the list of lists
    return [item for sublist in gathered_list for item in sublist]

def test_epoch(model, device, dataloader_q, dataloader_g, model_arch, writer, epoch, remove_junk=True, scaler=False,
               rank=0, world_size=1):
    model.eval()
    is_ddp = world_size > 1

    # Each process computes features for its subset of data
    qf_local, gf_local = [], []
    q_camids_local, g_camids_local = [], []
    q_vids_local, g_vids_local = [], []

    with torch.no_grad():
        # Disable tqdm progress bar on non-main processes
        query_pbar = tqdm(dataloader_q, desc='Query infer (%)', bar_format='{l_bar}{bar:20}{r_bar}',
                          disable=(rank != 0))
        for image, q_id, cam_id, view_id in query_pbar:
            image = image.to(device)
            # Forward pass logic is the same
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                _, _, ffs, _ = model(image, cam_id, view_id)

            ffs_cat = torch.cat([F.normalize(item) for item in ffs], dim=1)
            qf_local.append(ffs_cat.cpu())  # Move to CPU to save GPU memory before gathering
            q_vids_local.append(q_id)
            q_camids_local.append(cam_id)

        gallery_pbar = tqdm(dataloader_g, desc='Gallery infer (%)', bar_format='{l_bar}{bar:20}{r_bar}',
                            disable=(rank != 0))
        for image, g_id, cam_id, view_id in gallery_pbar:
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                _, _, ffs, _ = model(image, cam_id, view_id)

            ffs_cat = torch.cat([F.normalize(item) for item in ffs], dim=1)
            gf_local.append(ffs_cat.cpu())  # Move to CPU
            g_vids_local.append(g_id)
            g_camids_local.append(cam_id)

    ### DDP MODIFICATION ###: Gather results from all GPUs if in DDP mode
    if is_ddp:
        qf_all = gather_objects(qf_local, world_size)
        gf_all = gather_objects(gf_local, world_size)
        q_vids_all = gather_objects(q_vids_local, world_size)
        g_vids_all = gather_objects(g_vids_local, world_size)
        q_camids_all = gather_objects(q_camids_local, world_size)
        g_camids_all = gather_objects(g_camids_local, world_size)
    else:  # single GPU case
        qf_all, gf_all = qf_local, gf_local
        q_vids_all, g_vids_all = q_vids_local, g_vids_local
        q_camids_all, g_camids_all = q_camids_local, g_camids_local

    ### DDP MODIFICATION ###: Only rank 0 performs the final evaluation
    if rank == 0:
        # Concatenate all the gathered lists into single tensors
        qf = torch.cat(qf_all, dim=0)
        gf = torch.cat(gf_all, dim=0)
        q_vids = torch.cat(q_vids_all, dim=0).numpy()
        g_vids = torch.cat(g_vids_all, dim=0).numpy()
        q_camids = torch.cat(q_camids_all, dim=0).numpy()
        g_camids = torch.cat(g_camids_all, dim=0).numpy()

        # The rest of the evaluation logic is the same
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        distmat = torch.sqrt(distmat.clamp(min=1e-12)).cpu().numpy()

        cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)

        if writer:
            writer.write_scalars({"Accuraccy/CMC1": cmc[0], "Accuraccy/mAP": mAP}, epoch)

        return cmc, mAP
    else:
        # Other processes don't need to return evaluation results
        return None, None