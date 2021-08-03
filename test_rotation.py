import argparse
import json
import os
from pathlib import Path
from threading import Thread
from utils.rotation_nms import caculate_iou_rotation

import numpy as np
import torch
import yaml
from tqdm import tqdm
from detectron2._C import box_iou_rotated

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou,xywhtheta2polygons, non_max_suppression, non_max_suppression_rotation, scale_coords, scale_coords_polygon, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
# from rotation_nms import rotation_nms 


@torch.no_grad()
def test(data,
         weights=None,  # model.pt path(s)
         batch_size=32,  # batch size
         imgsz=640,  # inference size (pixels)
         conf_thres=0.001,  # confidence threshold
         iou_thres=0.6,  # NMS IoU threshold
         task='val',  # train, val, test, speed or study
         device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
         single_cls=False,  # treat as single-class dataset
         augment=False,  # augmented inference
         verbose=False,  # verbose output
         save_txt=False,  # save results to *.txt
         save_hybrid=False,  # save label+prediction hybrid results to *.txt
         save_conf=False,  # save confidences in --save-txt labels
         save_json=False,  # save a cocoapi-compatible JSON results file
         project='runs/test',  # save to project/name
         name='exp',  # save to project/name
         exist_ok=False,  # existing project/name ok, do not increment
         half=True,  # use FP16 half-precision inference
         model=None,
         dataloader=None,
         save_dir=Path(''),
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    is_coco = data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    # num_labels=0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []


    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t_ = time_synchronized()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_synchronized()
        t0 += t - t_

        # Run model
        out, train_out = model(img, augment=augment)  # inference and training outputs
        t1 += time_synchronized() - t
        # import pdb;pdb.set_trace()
        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        # import pdb;pdb.set_trace()
        targets[:, 2:] *= torch.Tensor([width, height, width, height,90.]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        out = non_max_suppression_rotation(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t2 += time_synchronized() - t

        # Statistics per image
        # import pdb;pdb.set_trace()
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1
            # num_labels+=nl
            gti=np.zeros(nl)
            
            if len(pred) == 0:
                if nl:
                    # groudtruth=np.hstack((groudtruth,gti))
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            # import pdb;pdb.set_trace()
            #stack tp score
            # tpi=np.zeros(len(pred))

            # score_pi=np.zeros(len(pred))
            # pred_clsi=np.zeros(len(pred))
            # score_pi=pred[:,-1]

            # # gti=np.zeros(nl)
            # target_clsi=np.zeros(nl)
            # Predictions
            # import pdb;pdb.set_trace()
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # import pdb;pdb.set_trace()
            scale_coords_polygon(img[si].shape[1:], predn[:], shapes[si][0], shapes[si][1])  # native-space pred
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                # import pdb;pdb.set_trace()
                
                # target boxes
                tboxes =labels[:, 1:]
                scale_coords_polygon(img[si].shape[1:], tboxes, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tboxes), 1))

                # Per target class
                # tp[i]
                # import pdb;pdb.set_trace()
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 6]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    if pi.shape[0]:
                        # import pdb;pdb.set_trace()
                        ious, i =box_iou_rotated(predn[pi,:5],tboxes[ti]).max(1)
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break 
                # import pdb;pdb.set_trace()
            # tp=np.hstack((tp,tpi))
            # confidence=np.hstack((confidence,score_pi))
            # groudtruth=np.hstack((groudtruth,gti))
            # pred_cls=np.hstack((pred_cls,pred_clsi))
            # target_cls=np.hstack((target_cls,target_clsi))


            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))

        # Plot images
        # if plots and batch_i < 3:
        #     f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
    # import pdb;pdb.set_trace()
    # Compute statistics
    # import pdb;pdb.set_trace()
    #tp=tp.reshape(len(tp),-1)
    # confidence=confidence.reshape(1,-1)
    # pred_cls=pred_cls.reshape(1,-1)
    # if len(tp) and tp.any():
    #     tp=tp.reshape(len(tp),-1)
    #     p, r, ap, f1, ap_class = ap_per_class(tp,confidence, pred_cls,target_cls,plot=plots, save_dir=save_dir, names=names)
        
    #     # import pdb;pdb.set_trace()
    #     ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    #     mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    #     nt = len(groudtruth)  # number of targets per class
    # import pdb;pdb.set_trace()
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # import pdb;pdb.set_trace()
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc) 
    else:
        nt = torch.zeros(1)
    # import pdb;pdb.set_trace()

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt, p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            test(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                 save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                               iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
