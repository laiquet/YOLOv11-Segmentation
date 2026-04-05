import os
from pathlib import Path
import torch
from utils import (smart_inference_mode, select_device,
                    check_img_size, colorstr, Profile, strip_optimizer)
from helpers import (process_mask, process_mask_upsample, non_max_suppression, masks2segments, scale_segments,
                     increment_path, colors, plot_masks, scale_masks, save_one_box, scale_boxes, scale_coords, scale_boxes_123, masks2segments_231, scale_coords_231, scale_masks_231, process_mask_231)
from dataloader import LoadImages, IMG_FORMATS
from dmb import AutoBackend, DetectMultiBackend
from helpers import Annotator
import cv2
import argparse

#Root directory
ROOT = os.getcwd()

@smart_inference_mode()
def run(
        weights=ROOT+'/train-seg/weights/best_spots.pt',  # model.pt path(s)
        source=ROOT+'/test/images',  # file/dir
        data=None,  # dataset.yaml path
        imgsz=(640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=300,  # maximum detections per image
        device=0,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT+'/runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        retina_mask=False
    ):  


    print(f'Conf:{conf_thres}')
    print(f'IOU:{iou_thres}')
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = AutoBackend(weights, 
                        device=device, 
                        dnn=dnn, 
                        data=data, 
                        fp16=half,
                        fuse=True, 
                        batch=None,
                        verbose=True)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load images
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    #Eval mode
    model.eval()

    #print(f'Predict Model:{model}')

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, dt = 0, (Profile(), Profile(), Profile())
    for path, im, im0s, s in dataset:
        print(f'Updated Image Shape (Predict ftn):{im.shape}')
        print(f'Original Image Shape (Predict ftn):{im0s.shape}')
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            print(len(im.shape))
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            preds = model(im, augment=False, visualize=visualize)

        # NMS
        with dt[2]:
            protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
            preds = non_max_suppression(preds[0], 
                                       conf_thres, 
                                       iou_thres, 
                                       classes, 
                                       agnostic_nms,
                                       max_det=max_det) 
            
            
        #print(f'protos:{protos}')
         # Process predictions
        for i, det in enumerate(preds):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            #start from here
            if len(det):
                if retina_mask: #Ignore for now casting issue with masks values! investigate further tomorrow!!!
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape)
                    masks = process_mask_upsample(protos[i], det[:, 6:], det[:, :4], im.shape[2:])  # HWC
                else:
                    masks = process_mask_231(protos[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
                    det[:, :4] = scale_boxes_123(im.shape[2:], det[:, :4], im0.shape) #updated

                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                """ ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data) """
                #test for 1
                """ for xeee in reversed(masks2segments_231(masks)):
                    print(f'One Mask:{xeee}')
                    meee = scale_coords_231(im.shape[1:], xeee, im0.shape, normalize=True)
                    print(f'One Converted Mask:{meee}')
                    break """
                
                # Segments
                #print(f'Masks (Predict ftn):{masks[0]}')
                if save_txt: #problematic section
                    segments = [scale_coords_231(im.shape[1:], x, im0.shape, normalize=True) for x in reversed(masks2segments_231(masks))]
                    #print(f'Inside Save Text Segments:{segments}')

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        #print('I am here baby')
                        segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                        #print(f'Predict seg:{segj}')
                        line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
                        #print(f'Predict line:{line}')
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #print(f'Predict xyxy:{xyxy}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        
            im0 = annotator.result()
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT+'/train-seg/weights/best_spots.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT+'/test/images', help='file/dir')
    parser.add_argument('--data', type=str, default=None, help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1080], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.75, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT+'/runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main():
    opt = parse_opt()
    run(**vars(opt))

if __name__=="__main__":
    main()