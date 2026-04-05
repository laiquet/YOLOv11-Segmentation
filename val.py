import os
import yaml
from datetime import datetime
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP #Check the version for torch.__version__, '1.11.0' with static_graph=True or above
from loss import v8SegmentationLoss 
from callbacks import Callbacks
import torch.nn.functional as F
from utils import ( smart_inference_mode, de_parallel, select_device,
                    check_img_size, colorstr, Profile, yaml_load)
from helpers import (xyxy2xywh,mask_iou,box_iou, process_mask, ConfusionMatrix, Metrics, tqdm, non_max_suppression, scale_coords, 
                   xywh2xyxy, plot_images_and_masks, output_to_target, ap_per_class_box_and_mask, increment_path, scale_boxes, box_iou_yolov11)
from dataloader import create_dataloader
from dmb import AutoBackend
import json

#Root directory
ROOT = os.getcwd()
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
    else:  # boxes
        #iou = box_iou(labels[:, 1:], detections[:, :4]) #bbox_iou_yolov11
        iou = box_iou_yolov11(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data=ROOT+'/data1/data.yaml',
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT+'/runs/val-seg',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        classes=None,
        agnostic_nms=False,
        plots=True,
        overlap=False,
        mask_downsample_ratio=1,
        compute_loss=None,
        hyp=None
        #callbacks=Callbacks(),
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        #load hyperparameters
        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict

        # Load model
        model = AutoBackend(weights, 
                        device=device, 
                        dnn=dnn, 
                        data=data, 
                        fp16=half,
                        fuse=True, 
                        batch=None,
                        verbose=True)
        stride, pt = model.stride, model.pt
        model.model.hyp = hyp
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA

    if data is not None:
        data = yaml_load(data)

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(batch_size, 3, *imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz[0],
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '),
                                       overlap_mask=overlap,
                                       mask_downsample_ratio=mask_downsample_ratio)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)", "Mask(P", "R",
                                  "mAP50", "mAP50-95)")
    dt = Profile(), Profile(), Profile()
    metrics = Metrics()
    loss = torch.zeros(4, device=device)
    jdict, stats = [], []
    # callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        # callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device)
            masks = masks.float()
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            out = model(im)  # if training else model(im, augment=augment, val=True)  # inference, loss

        # Loss
        if compute_loss is None:
            compute_loss = v8SegmentationLoss(model.model, overlap=overlap)
        loss += compute_loss(batch_i, out, targets, masks)[1]

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        with dt[2]:
            protos = out[1][-1] if isinstance(out[1], tuple) else out[1]
            preds = non_max_suppression(out[0], 
                                        conf_thres, 
                                        iou_thres, 
                                        classes, 
                                        agnostic_nms,
                                        nc=nc, 
                                        max_det=max_det) 

        # Metrics
        plot_masks = []  # masks for plotting
        for si, (pred, proto) in enumerate(zip(preds, protos)):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Masks
            midx = [si] if overlap else targets[:, 0] == si
            gt_masks = masks[midx]
            pred[:,:4] = xywh2xyxy(pred[:,:4])
            pred_masks = process_mask(proto, pred[:, 6:], pred[:, :4], im[si].shape[1:], shape)

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            #predn[:,:4] = xywh2xyxy(predn[:,:4])
            predn[:, :4] = scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                tbox = scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct_bboxes = process_batch(predn, labelsn, iouv)
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls)

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if plots and batch_i < 3:
                plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')


        # Plot images
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            plot_images_and_masks(im, targets, masks, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)
            plot_images_and_masks(im, output_to_target(preds, max_det=15), plot_masks, paths,
                                  save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        # callbacks.run('on_val_batch_end')

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 8  # print format
    print(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
    if nt.sum() == 0:
        print(f'WARNING: no labels found in {task} set, can not compute metrics without labels ⚠️')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            print(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # callbacks.run('on_val_end')

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            results = []
            for eval in COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'segm'):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # img ID to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])  # update results (mAP@0.5:0.95, mAP@0.5)
            map_bbox, map50_bbox, map_mask, map50_mask = results
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT+'/data1/data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT+'/train-seg/weights/best.pt', help='initial weights path')
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=[640], help='train, val image size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', type=str, default='val', help='train, val, test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single_cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--verbose', default=False, action='store_true', help='verbose output')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save_json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT+'/runs/val-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', default=False, action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--model', type=str, default=None, help='')
    parser.add_argument('--dataloader', type=str, default=None, help='')
    parser.add_argument('--save_dir', default=ROOT+'/val-seg', help='save model weights and stuff')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--plots', default=True, action='store_true', help='save plots')
    parser.add_argument('--overlap', type=bool, default=True, help='validate with masks overlap')
    parser.add_argument('--mask_downsample_ratio', type=float, default=1, help='Downsampling masks ratio')
    parser.add_argument('--compute_loss', type=str, default=None, help='')
    parser.add_argument('--hyp', type=str, default=ROOT+'/data/hyps/hyp.scratch-high.yaml', help='model hyperparameters')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main():
    opt = parse_opt()
    run(**vars(opt))

if __name__=="__main__":
    main()