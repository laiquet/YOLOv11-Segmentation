import argparse
import math
import os
import random
import torch.nn.functional as F
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP #Check the version for torch.__version__, '1.11.0' with static_graph=True or above
import yaml
from torch.optim import lr_scheduler
from yolo import SegmentationModel
from loss import v8SegmentationLoss
from callbacks import Callbacks
from utils import (check_img_size, ModelEMA, colorstr, de_parallel, strip_optimizer, attempt_load, select_device)
from helpers import (smart_optimizer, one_cycle, check_anchors, labels_to_class_weights, EarlyStopping, labels_to_image_weights,
                     plot_images_and_masks, fitness, KEYS, plot_results_with_masks, intersect_dicts, model_info, build_optimizer)
from dataloader import create_dataloader
from val import run as validaterun
from tqdm import tqdm
from numpy.core.multiarray import _reconstruct  # import the function
from logger import LOGGER

#Root directory
ROOT = os.getcwd()
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# Register the global so that it's allowed during loading
torch.serialization.add_safe_globals([SegmentationModel])
#torch.serialization.add_safe_globals([_reconstruct])

def check_amp(model):
    """Check PyTorch Automatic Mixed Precision (AMP) functionality. Return True on correct operation."""
    device = next(model.parameters()).device
    if device.type == 'cpu':
        return False  # AMP disabled on CPU
    try:
        # Simple AMP check: run a small forward pass with autocast
        with torch.amp.autocast(device_type='cuda'):
            x = torch.zeros(1, 3, 32, 32, device=device)
            _ = model(x)
        LOGGER.info('AMP: checks passed ✅')
        return True
    except Exception as e:
        LOGGER.warning(f'AMP: checks failed ❌, disabling Automatic Mixed Precision. Error: {e}')
        return False

def train(hyp, opt, device, callbacks):
    # Usage:
        #   .....

    # Training parameters setup
    save_dir, epochs, batch_size, weights, single_cls, data, cfg, resume, noval, nosave, workers, freeze, mask_ratio = \
    Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.data, opt.cfg, \
    opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.mask_ratio

    # Getting the training device
    cuda = device.type != 'cpu'
    overlap = not opt.no_overlap

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    # Loading data stuff
    # Read data yaml
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Configs
    train_path, val_path = data['train'], data['val']
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data['names']) != 1 else data['names']  # class names
    plots = not opt.noplots

    # Create model
    model = SegmentationModel(cfg, ch=3, nc=nc).to(device)  # create

    # Load checkpoint with module remapping (pretrained .pt files reference ultralytics.* paths)
    import sys, types
    _remap = {
        'ultralytics.nn.tasks': 'yolo',
        'ultralytics.nn.modules.block': 'common_layers',
        'ultralytics.nn.modules.conv': 'common_layers',
        'ultralytics.nn.modules.head': 'yolo',
        'ultralytics.nn.modules.transformer': 'common_layers',
        'ultralytics.nn.modules.utils': 'common_layers',
        'ultralytics.nn.modules': 'common_layers',
        'ultralytics.utils.tal': 'loss',
    }
    # Temporarily inject fake ultralytics modules that redirect to local ones
    _injected = []
    for fake_mod, real_mod in _remap.items():
        if fake_mod not in sys.modules:
            # Create parent packages if needed (e.g. ultralytics, ultralytics.nn)
            parts = fake_mod.split('.')
            for i in range(1, len(parts) + 1):
                parent = '.'.join(parts[:i])
                if parent not in sys.modules:
                    sys.modules[parent] = types.ModuleType(parent)
                    _injected.append(parent)
            # Point the fake module to the real local module
            real = __import__(real_mod)
            sys.modules[fake_mod] = real
            if fake_mod not in _injected:
                _injected.append(fake_mod)

    ckpt = torch.load(weights, map_location='cpu', weights_only=False)

    # Clean up injected modules
    for mod in _injected:
        sys.modules.pop(mod, None)
    exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(f'Loaded Model:{model_info(model)}')
    model.nc = nc
    amp = check_amp(model)  # check AMP

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # NOTE: Train dataloader is created after optimizer/scheduler setup (below)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay = hyp['weight_decay'] * batch_size * accumulate / nbs  # scale weight_decay

    # LR schedule function (scheduler created after optimizer below)
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: max(1 - x / epochs, 0) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # SyncBatchNorm (must be before DDP wrapping)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.')
        model = torch.nn.DataParallel(model)

    # Create train dataloader
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == 'val' else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr('train: '),
        shuffle=True,
        mask_downsample_ratio=mask_ratio,
        overlap_mask=overlap,
    )

    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Validation dataloader (only on main process)
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            mask_downsample_ratio=mask_ratio,
            overlap_mask=overlap,
            prefix=colorstr('val: ')
        )[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Scale loss hyperparameters by detection layers, classes, and image size
    nl = de_parallel(model).model[-1].nl  # number of detection layers
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['dfl'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.hyp = hyp
    model.names = names

    # Build optimizer now that dataloader exists
    iterations = math.ceil(len(train_loader.dataset) / max(batch_size, nbs)) * epochs
    optimizer = build_optimizer(hyp,
                                model,
                                opt.optimizer,
                                hyp['lr0'],
                                hyp['momentum'],
                                weight_decay,
                                iterations=iterations)
    LOGGER.info(f'Optimizer: {type(optimizer).__name__} with {len(optimizer.param_groups)} param groups')
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Training loop setup
    best_fitness = 0.0
    start_epoch = 0
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # warmup iterations
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = v8SegmentationLoss(model, overlap=overlap)

    # Resume from checkpoint
    if resume and last.exists():
        ckpt_resume = torch.load(last, map_location='cpu', weights_only=False)
        start_epoch = ckpt_resume.get('epoch', -1) + 1
        best_fitness = ckpt_resume.get('best_fitness', 0.0)
        if ckpt_resume.get('optimizer'):
            optimizer.load_state_dict(ckpt_resume['optimizer'])
        if ema and ckpt_resume.get('ema'):
            ema.ema.load_state_dict(ckpt_resume['ema'].float().state_dict())
            ema.updates = ckpt_resume.get('updates', 0)
        LOGGER.info(f'Resumed training from epoch {start_epoch}, best_fitness={best_fitness:.4f}')
        del ckpt_resume

    scheduler.last_epoch = start_epoch - 1

    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs (from epoch {start_epoch})...')

    optimizer.zero_grad(set_to_none=True)  # zero any resumed gradients
    for epoch in range(start_epoch, epochs):

        model.train()

        # Shuffle for DDP
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        mloss = torch.zeros(4, device=device)  # mean losses

        # --- CREATE A SINGLE PROGRESS BAR ONLY FOR RANK IN {-1, 0} ---
        if RANK in {-1, 0}:
            # Show tqdm progress bar
            pbar = tqdm(enumerate(train_loader),
                        total=nb,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                        desc=f"Epoch {epoch}/{epochs - 1}")
            LOGGER.info(('\n' + '%11s' * 8) %
                        ('Epoch', 'GPU_mem', 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'Instances', 'Size'))
        else:
            # Normal enumeration without tqdm
            pbar = enumerate(train_loader)

        for i, (imgs, targets, paths, _, masks) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr
                    x['lr'] = np.interp(ni,
                                        xi,
                                        [hyp['warmup_bias_lr'] if j == 0 else 0.0,
                                         x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs)) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward + loss
            with torch.amp.autocast(device_type='cuda', enabled=amp):
                pred = model(imgs)
                loss, loss_items = compute_loss(i, pred, targets.to(device), masks=masks.to(device).float())
                if RANK != -1:
                    loss *= WORLD_SIZE  # DDP
                if opt.quad:
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Logging on main process only
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                if hasattr(pbar, 'set_description'):
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 6) %
                                         (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

                # Optionally plot first batches
                if ni < 3 and plots:
                    # Upscale masks for visualization if needed
                    if mask_ratio != 1:
                        masks_show = F.interpolate(masks[None].float(),
                                                   (imgsz, imgsz),
                                                   mode="bilinear",
                                                   align_corners=False)[0]
                    else:
                        masks_show = masks
                    plot_images_and_masks(imgs, targets, masks_show, paths, save_dir / f"train_batch{ni}.jpg")

        # Scheduler
        scheduler.step()

        # Validation
        if RANK in {-1, 0}:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:
                results, _, _ = validaterun(
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            half=amp,
                                            model=ema.ema,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            plots=False,
                                            classes=None,
                                            agnostic_nms=False,
                                            compute_loss=compute_loss,
                                            mask_downsample_ratio=mask_ratio,
                                            overlap=overlap)

            # Log results to CSV
            results_file = save_dir / 'results.csv'
            header = ('epoch', 'train/box_loss', 'train/seg_loss', 'train/cls_loss', 'train/dfl_loss',
                      'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP_0.5(B)', 'metrics/mAP_0.5:0.95(B)',
                      'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP_0.5(M)', 'metrics/mAP_0.5:0.95(M)',
                      'val/box_loss', 'val/seg_loss', 'val/cls_loss', 'val/dfl_loss')
            n_cols = len(header)
            vals = [epoch] + list(mloss.cpu().numpy()) + list(results[:8]) + list(results[8:])
            if not results_file.exists():
                with open(results_file, 'w') as f:
                    f.write(', '.join(['%23s'] * n_cols) % header + '\n')
            with open(results_file, 'a') as f:
                f.write(', '.join(['%23.5g'] * n_cols) % tuple(vals) + '\n')

            # Check best fitness
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not nosave) or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()
                }
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch_{epoch}.pt')
                    LOGGER.info(f'Saved checkpoint: {w / f"epoch_{epoch}.pt"}')
                del ckpt

            # EarlyStopping across processes
            if RANK != -1:
                broadcast_list = [stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                if RANK != 0:
                    stop = broadcast_list[0]
            if stop:
                break

    # Training completed
    if RANK in {-1, 0}:
        print(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in [last, best]:
            if f.exists():
                strip_optimizer(f)
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validaterun(
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.60,
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=False,
                        verbose=True,
                        plots=plots,
                        classes=None,
                        agnostic_nms=False,
                        compute_loss=compute_loss,
                        mask_downsample_ratio=mask_ratio,
                        overlap=overlap
                    )
        if plots:
            try:
                plot_results_with_masks(file=save_dir / 'results.csv')
            except Exception as e:
                LOGGER.warning(f'Could not plot results: {e}')
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT+'/pretrained/yolo11x-seg.pt', help='initial weights path') #ROOT+'/pretrained/yolo11x-seg.pt'
    parser.add_argument('--cfg', type=str, default=ROOT+'/model/yolo11x-seg.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT+'/data1/data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT+'/data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT+'/runs/trainseg', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--save_dir', default=ROOT+'/train-seg', help='save model weights and stuff')

    # Instance Segmentation Args
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main():
    callbacks=Callbacks()
    opt = parse_opt()
    device = select_device(opt.device, batch_size=opt.batch_size)
    train(opt.hyp, opt, device, callbacks)

if __name__=="__main__":
    main()