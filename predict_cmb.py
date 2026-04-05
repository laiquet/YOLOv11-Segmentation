import os
from pathlib import Path
import torch
from utils import (smart_inference_mode, select_device,
                    check_img_size, colorstr, Profile, strip_optimizer, sorting_custom, plot_lineplot, plot_swarmplot, plot_boxplot, plot_scatterplot, plot_violinplot)
from helpers import (process_mask, non_max_suppression, masks2segments, scale_coords,
                     increment_path, colors, plot_masks, scale_masks, save_one_box, scale_boxes, xywh2xyxy, torchvision)
from dataloader import LoadImages, IMG_FORMATS
from dmb import AutoBackend
from helpers import Annotator
import cv2
import argparse
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Root directory
ROOT = os.getcwd()

@smart_inference_mode()
def run(
        weights1=ROOT+'/train-seg/weights/best_yolov11_spots.pt',  # model.pt path(s)
        weights2=ROOT+'/train-seg/weights/best_yolov8_modified_spots.pt',
        source=ROOT+'/test/images',  # file/dir
        data=None,  # dataset.yaml path
        imgsz=(640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=300,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_custom_plots=False, #save custom plots
        save_stress_exp_data = False, #save stress experiment data in csv format for statistical analysis
        save_stress_exp_plots = False, #save stress experiment data in csv format for statistical analysis
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
    ):  

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    #Custom plots
    if save_custom_plots:
        (save_dir / 'custom plots').mkdir(parents=True, exist_ok=True)  # make dir custom plots
    #CSV data
    if save_stress_exp_data:
        (save_dir / 'Stat data').mkdir(parents=True, exist_ok=True)  # make dir stat data

    #CSV data
    if save_stress_exp_plots:
        (save_dir / 'Stat plots' / 'lineplots').mkdir(parents=True, exist_ok=True)  # make dir lineplots
        (save_dir / 'Stat plots' / 'swarmplots').mkdir(parents=True, exist_ok=True)  # make dir swarmplots
        (save_dir / 'Stat plots' / 'boxplots').mkdir(parents=True, exist_ok=True)  # make dir boxplots
        (save_dir / 'Stat plots' / 'scatterplots').mkdir(parents=True, exist_ok=True)  # make dir scatterplots
        (save_dir / 'Stat plots' / 'violinplots').mkdir(parents=True, exist_ok=True)  # make dir violinplots

    # Load model
    device = select_device(device)
    #Yolov11
    model1 = AutoBackend(weights1, 
                        device=device, 
                        dnn=dnn, 
                        data=data, 
                        fp16=half,
                        fuse=True, 
                        batch=None,
                        verbose=True)
    #Yolov8
    model2 = AutoBackend(weights2, 
                        device=device, 
                        dnn=dnn, 
                        data=data, 
                        fp16=half,
                        fuse=True, 
                        batch=None,
                        verbose=True)

    #Combined - Since both of the model will have same strides for the image and names for the labels
    stride, names, pt = model1.stride, model1.names, model1.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Load images
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    #Eval mode
    model1.eval() #yolov11
    model2.eval() #yolov8

    # Run inference
    model1.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup yolov11
    model2.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup yolov8

    seen, dt = 0, (Profile(), Profile(), Profile())

    #Save stress experiment data
    if save_stress_exp_data:
        global_signature = {}

    if save_stress_exp_plots: #there you go another if - DO SOMETHING ABOUT IT!
        global_signature_2={}

    for path, im, im0s, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model1.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            preds1 = model1(im, augment=False, visualize=visualize) #yolov11 predictions
            preds2 = model2(im, augment=False, visualize=visualize) #yolov8 predictions

        # NMS
        with dt[2]:
            #Yolov11 post-processing
            protos1 = preds1[1][-1] if isinstance(preds1[1], tuple) else preds1[1]
            preds1 = non_max_suppression(preds1[0], 
                                        conf_thres, 
                                        iou_thres, 
                                        classes, 
                                        agnostic_nms,
                                        nc=len(list(names.keys())), 
                                        max_det=max_det) 
            #Yolov8 post-processing
            protos2 = preds2[1][-1] if isinstance(preds2[1], tuple) else preds2[1]
            preds2 = non_max_suppression(preds2[0], 
                                        0.1, 
                                        iou_thres, 
                                        classes, 
                                        agnostic_nms,
                                        nc=len(list(names.keys())), 
                                        max_det=max_det)
            
        #Process Yolov11 predictions
        for i, (det1, det2) in enumerate(zip(preds1, preds2)):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            cplots_path = str(save_dir / 'custom plots' / p.name) #custom plots
            sedata_path = str(save_dir / 'Stat data' / 'data.csv') #custom stress experiment data
            lplots = str(save_dir/ 'Stat plots' / 'lineplots') #custom line plots 
            swplots = str(save_dir/ 'Stat plots' / 'swarmplots') #custom swarm plots
            bplots = str(save_dir/ 'Stat plots' / 'boxplots') #custom box plots
            splots = str(save_dir/ 'Stat plots' / 'scatterplots') #custom scatter plots
            vplots = str(save_dir/ 'Stat plots' / 'violinplots') #custom violin plots
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if save_custom_plots:
                imcp = im0.copy() 
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det1) or len(det2):
                det1[:,:4] = xywh2xyxy(det1[:,:4])
                det1[:, :4] = scale_boxes(im.shape[2:], det1[:, :4], im0.shape)
                masks1 = process_mask(protos1[i], det1[:, 6:], det1[:, :4], im.shape[2:], im0.shape)  # HWC
                det2[:,:4] = xywh2xyxy(det2[:,:4])
                det2[:, :4] = scale_boxes(im.shape[2:], det2[:, :4], im0.shape)
                masks2 = process_mask(protos2[i], det2[:, 6:], det2[:, :4], im.shape[2:], im0.shape)  # HWC

                masks_interim = torch.cat((masks1,masks2), 0)
                boxes_interim = torch.cat((det1[:, :4],det2[:, :4]), 0)
                scores_interim = torch.cat((det1[:, 4],det2[:, 4]), 0)
                all_boxxes = torch.cat((det1,det2), 0)

                idxs = torchvision.ops.nms(boxes_interim, scores_interim, 0.45)  # NMS

                masks = masks_interim[idxs]
                bboxes = boxes_interim[idxs]
                scores = scores_interim[idxs]
                det = all_boxxes[idxs]

                # Segments
                if save_txt: 
                    segments = reversed(masks2segments(masks))
                    segments = [scale_coords(im.shape[2:], x, im0.shape, normalize=True) for x in segments]

                # Print results need to update this code for
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im_masks, im0.shape)  # scale to original h, w

                # Write results
                spots = 0
                if save_custom_plots:
                    fig, ax = plt.subplots(1)
                    ax.imshow(imcp)

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                        pixel_vertices = [int(segj[j] * im0.shape[1] if j % 2 == 0 else segj[j] * im0.shape[0]) for j in range(len(segj))]
                        #Gray scale signature analysis (vectorized)
                        mask = Image.new('L', (im0.shape[1], im0.shape[0]), 0)
                        ImageDraw.Draw(mask).polygon(pixel_vertices, outline=1, fill=1)
                        mask_np = np.array(mask, dtype=np.uint8)
                        img_compt = np.array(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
                        seg_spot = np.bitwise_and(img_compt, mask_np * 255)
                        seg_spot = seg_spot / 255.0
                        xs, ys = np.nonzero(seg_spot)

                        # Vectorized binning using np.digitize
                        pixel_vals = seg_spot[xs, ys]
                        bin_edges = np.arange(0.0, 1.1, 0.1)
                        bin_indices = np.digitize(pixel_vals, bin_edges, right=True)
                        bin_indices = np.clip(bin_indices, 1, 10)

                        sig_lists = []
                        pix_counts = []
                        for b in range(1, 11):
                            bin_mask = bin_indices == b
                            vals = pixel_vals[bin_mask]
                            sig_lists.append(vals)
                            pix_counts.append(len(vals))

                        sig_means = [np.array(s).mean() if len(s) > 0 else float('nan') for s in sig_lists]

                        #line with class label, polycords(1...n), confidence, pixel bins mean values(1..10), pixel bins total pixel count(1...10)
                        line_wpinf = (cls, *segj, conf, *sig_means, *pix_counts) if save_conf else (cls, *segj, conf)
                        
                        #line with class label, polycords(1...n), confidence
                        line = (cls, *segj, conf) if save_conf else (cls, *segj) 
                        
                        #For corrections later on in RF - no pixel info included
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        
                        #With pixel info included for later test and stuff
                        with open(f'{txt_path}_wpinfo.txt', 'a') as f:
                            f.write(('%g ' * len(line_wpinf)).rstrip() % line_wpinf + '\n')


                    if save_stress_exp_data or save_stress_exp_plots:
                        fid = p.name.split('_')[1]
                        day = p.name.split('_')[3][-1:]
                        side = p.name.split('_')[5]

                    if save_stress_exp_data:
                        pixel_values = np.concatenate(sig_lists)
                        pixel_values_clean = pixel_values[~np.isnan(pixel_values)]
                        pixel_values_mean = pixel_values_clean.mean() if len(pixel_values_clean) > 0 else float('nan')
                        pixel_count = sum(pix_counts)
                        global_signature[f'{p.name.split(".")[0]}_spot_{j}']=np.array([fid,
                                                                                      day,
                                                                                      side,
                                                                                      j,
                                                                                      pixel_values_mean,
                                                                                      pixel_count])
                    
                    if save_stress_exp_plots:
                        global_signature_2[f'{p.name.split(".")[0]}_spot_{j}'] = np.array([fid, day, side, j,
                                                *[np.array(s) for s in sig_lists]], dtype=object)
                        
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    if save_custom_plots:
                        #Create a polygon patch from the vertices and add it to the plot
                        polygon = patches.Polygon(xy=np.array(pixel_vertices).reshape(-1, 2),
                                                    closed=True,
                                                    edgecolor='green',
                                                    facecolor='none',
                                                    linewidth=1)
                        ax.add_patch(polygon)
                        ax.axis('off')  #Correct axis indexing and visibility handling
                        spots+=1
                        #Display the plot
                        plt.title(p.name.split('.')[0])
                        spots_leg = patches.Patch(color='green', label=f'Spots ({spots})')
                        legend_handles = [spots_leg]
                        plt.legend(handles=legend_handles, loc="upper left")
                        
            im0 = annotator.result()
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
            
            # Save custom plots 
            if save_custom_plots:
                plt.savefig(cplots_path, dpi=1200)
                plt.close(fig)  # prevent memory leak

        # Print time (inference-only)
        print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if save_stress_exp_data:
        sorted_data = dict(sorted(global_signature.items(), key=lambda x: sorting_custom(x[0])))
        data_fish = pd.DataFrame.from_dict(sorted_data, 
                                           orient='index', 
                                           columns=['FISH_ID', 'DAY', 'SIDE', 'spot','GSLM','GSLC'])
        data_fish.to_csv(sedata_path, index=False)

    if save_stress_exp_plots:
        sorted_data_2 = dict(sorted(global_signature_2.items(), key=lambda x: sorting_custom(x[0])))
        data4plots = pd.DataFrame.from_dict(sorted_data_2, 
                                    orient='index', 
                                    columns=['FISH_ID','DAY','SIDE','spot', 'level_1_m', 'level_2_m', 'level_3_m', 'level_4_m', 'level_5_m',
                                             'level_6_m', 'level_7_m', 'level_8_m', 'level_9_m', 'level_10_m'])
        data_pixels_1 = data4plots.reset_index()
        data_pixels_1 = data_pixels_1.drop(columns='index')
        fishids = np.unique(data_pixels_1.FISH_ID.values)

        #Sample 10 DataFrames
        for fid in fishids:
            df_sample = data_pixels_1[data_pixels_1.FISH_ID==f'{fid}']
            l1,l2,l3,l4,l5,l6,l7,l8,l9,l10 = df_sample['level_1_m'],df_sample['level_2_m'],df_sample['level_3_m'],df_sample['level_4_m'],df_sample['level_5_m'],df_sample['level_6_m'],df_sample['level_7_m'],df_sample['level_8_m'],df_sample['level_9_m'],df_sample['level_10_m'],
            cols = np.vstack((l1.values,l2.values,l3.values,l4.values,l5.values,l6.values,l7.values,l8.values,l9.values,l10.values))
            spots = {}
            for i in range(cols.shape[1]):
                sub_cols = []
                for j in range(cols.shape[0]):
                    sub_cols.extend(cols[j][i])
                spots[i] = np.array(sub_cols)
            df_sample['combined_lvls'] = spots.values()
            df_sample = df_sample.drop(columns=df_sample.columns[4:14])
            df_sample['means'] = [x.mean() for x in df_sample['combined_lvls'].values]
            df_sample['count'] = [len(x) for x in df_sample['combined_lvls'].values]

            L_D1 = df_sample[(df_sample['DAY'] == '1') & (df_sample['SIDE'] == 'L')]
            L_D2 = df_sample[(df_sample['DAY'] == '2') & (df_sample['SIDE'] == 'L')]
            R_D1 = df_sample[(df_sample['DAY'] == '1') & (df_sample['SIDE'] == 'R')]
            R_D2 = df_sample[(df_sample['DAY'] == '2') & (df_sample['SIDE'] == 'R')]

            L_D1_cmb = L_D1['combined_lvls'].values
            L_D2_cmb = L_D2['combined_lvls'].values
            R_D1_cmb = R_D1['combined_lvls'].values
            R_D2_cmb = R_D2['combined_lvls'].values

            L_D1_cmb_arr = []
            L_D2_cmb_arr = []
            R_D1_cmb_arr = []
            R_D2_cmb_arr = []
            for l1,l2,r1,r2 in zip(L_D1_cmb,L_D2_cmb,R_D1_cmb,R_D2_cmb):
                L_D1_cmb_arr.extend(l1)
                L_D2_cmb_arr.extend(l2)
                R_D1_cmb_arr.extend(r1)
                R_D2_cmb_arr.extend(r2)

            df_plot = pd.DataFrame()
            df_plot['FISH_ID'] = [10,10,10,10]
            df_plot['SIDE'] = ['L','L','R','R']
            df_plot['DAY'] = ['1','2','1','2']
            df_plot['GSLS'] = [np.array(L_D1_cmb_arr),np.array(L_D2_cmb_arr),np.array(R_D1_cmb_arr),np.array(R_D2_cmb_arr)]
            df_plot['GSLS_Mean'] = [np.array(L_D1_cmb_arr).mean(),np.array(L_D2_cmb_arr).mean(),np.array(R_D1_cmb_arr).mean(),np.array(R_D2_cmb_arr).mean()]


            # Explode the GSLS column
            df_exploded = df_plot.explode('GSLS')

            # Convert GSLS to numeric
            df_exploded['GSLS'] = df_exploded['GSLS'].astype(float)


            #Plots
            plot_lineplot(df_sample, fid, lplots)
            plot_swarmplot(df_sample, fid, swplots)
            plot_boxplot(df_sample, fid, bplots)
            plot_scatterplot(df_sample, fid, splots)
            plot_violinplot(df_exploded, fid, vplots)

    if update:
        strip_optimizer(weights1[0])
        strip_optimizer(weights2[0]) # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights1', nargs='+', type=str, default=ROOT+'/train-seg/weights/best_yolov11_spots.pt', help='model path(s)')
    parser.add_argument('--weights2', nargs='+', type=str, default=ROOT+'/train-seg/weights/best_yolov8_modified_spots.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT+'/samples', help='file/dir')
    parser.add_argument('--data', type=str, default=None, help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1080], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.30, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-custom-plots', action='store_true', help='save custom plots')
    parser.add_argument('--save-stress-exp-data', action='store_true', help='save stress experiment csv data for stat analysis') #save_stress_exp_data
    parser.add_argument('--save-stress-exp-plots', action='store_true', help='save stress experiment plots') #save_stress_exp_data
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