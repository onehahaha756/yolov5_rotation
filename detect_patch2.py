import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import cv2,glob,os,pickle
import torch,math,json
import torch.backends.cudnn as cudnn

import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, non_max_suppression_rotation, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.remote_utils import crop_xyxy2ori_xyxy,nms,draw_clsdet,draw_clsdet_rotation,rboxes2points
from utils.eval_rotation import casia_eval
import gdal 
from detectron2.layers import nms_rotated
multi_img_type=['*.jpg','*.png','*.tif','*.tiff']
# multi_img_type=['*PAN.tif']# remote origin image
#def work_dirs(data_dir):
import math
CLASSES=['A','B','C','D','E','F','G','H','I','J','K']
CLASSES=['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court', 'basketball-court',
               'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor',
               'swimming-pool', 'helicopter',
               'container-crane','airport','helipad']
CLASSES=['ship']

#def 
@torch.no_grad()
def detect_patch(weights='yolov5s.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           clssname='',
           imgsz=640,  # inference size (pixels)
           conf_thres=0.01,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           nosave=False,  # do not save images/videos
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           save_pkl=False,  # save results to *.txt
           save_json=None,  # save json  labels
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           exist_ok=False,  # existing project/name ok, do not increment
           half=False,  # use FP16 half-precision inference
           ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_pkl else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    print('conf thres : {} \nnms iou thres:  {}'.format(conf_thres,iou_thres))
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    print('weight path: ',weights)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    imglist=[]
    for img_type in multi_img_type:
        imglist+=glob.glob(os.path.join(source,img_type))
    # import pdb;pdb.set_trace()
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    vis_dir=osp.join(save_dir,'vis_results')
    if not osp.exists(vis_dir):
        os.mkdir(vis_dir)

    det_results={}
    save_results=[]
    dataset=LoadImages(source,img_size=imgsz,stride=stride)
    for path, img, im0s, vid_cap in dataset:
        # import pdb;pdb.set_trace()
        ts = time.time()
        basename=os.path.splitext(os.path.basename(path))[0]

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        tload=time.time()
        print('load time: {}s'.format(tload-ts))
        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        tinfer=time.time()
        print(f'infer time {tinfer-tload}s')
        pred = non_max_suppression_rotation(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        tnms=time.time()
        print(f'nms time {tnms-tinfer}s')
        pred=pred[0]
        #import pdb;pdb.set_trace()

        if save_json:
            img_result=rboxes2points(pred,clssname)
            save_result={}
            save_result['image_name']=osp.basename(path)
            save_result['labels']=img_result
            save_results.append(save_result)            
        if save_img:
            show_img=img.copy()
            show_img2=draw_clsdet_rotation(show_img,pred,clssname,conf_thres) 
            save_path = osp.join(vis_dir,'{}.jpg'.format(basename))  
            cv2.imwrite(save_path,show_img2)  
            print(f'{save_path} saved!')
        pred=[pd[:-1].cpu().numpy().tolist()+[clssname[int(pd[-1])]] for pd in pred]
        det_results[basename]=pred


        #print(f'{i}/{len(imglist)} ({time.time()-ts:.3f}s ETA: {(time.time()-ts)*(len(imglist)-i):.3f}s)')

    det_file=os.path.join(save_dir,'results.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(det_results, f, pickle.HIGHEST_PROTOCOL)
        print(f'save {det_file} successed!')
    print(f'Done. ({time.time() - t0:.3f}s)')
    return det_file

def save_patch_results(det_path):
    dirname=osp.dirname(det_path)
    patch_results_dir=osp.join(dirname,'patch_results')
    if osp.exists(patch_results_dir):
        shutil.rmtree(patch_results_dir)
        os.mkdir(patch_results_dir)
    else:
        os.mkdir(patch_results_dir)
    det_file=open(det_path,'rb')
    det_dict=pickle.load(det_file)
    import pdb;pdb.set_trace()
    for imgname in det_dict.keys():
        img_results=det_dict[imgname]
        for result in img_results:
            # import pdb;pdb.set_trace()
            xc, yc, w, h, ag,score,classname=result
            bbox_points=cv2.boxPoints(((xc,yc),(w,h),ag)).astype(np.int32)
            bbox_points=bbox_points.reshape(1,-1)[0].tolist()
            bbox_points=[max(0,x) for x in bbox_points]
            x1,y1,x2,y2,x3,y3,x4,y4=bbox_points
            
            save_path=osp.join(patch_results_dir,'Task1_{}.txt'.format(classname))
            if osp.exists(save_path):
                f=open(save_path,'a',encoding='utf-8')
            else:
                f=open(save_path,'a',encoding='utf-8')
            f.write('{} {:.3f} {} {} {} {} {} {} {} {}\n'.format(imgname,score,x1,y1,x2,y2,x3,y3,x4,y4))
            f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--dataset', type=str, default='data/dotav2.yaml', help='dataset_config')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--overlap', type=int, default=100, help='sub image overlap size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--remote', action='store_true', help='inference big remote images')    
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--eval', default=None, help='just eval')
    parser.add_argument('--save-pkl', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_json', type=str,default='submit/aircraft_results.json' ,help='save json results')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)

    import yaml 
    dataconfig=open(opt.dataset,'r',encoding='utf-8')
    dataset=yaml.load(dataconfig)
    classnames=dataset['names']

    test_images=dataset['test_images']
    test_labels=dataset['test_labels']
    clssname=dataset['names']
    # import pdb;pdb.set_trace()
    if not opt.eval:
        det_path=detect_patch(opt.weights,test_images,clssname,opt.imgsz,
                    opt.conf_thres,opt.iou_thres,opt.nosave,opt.project,opt.name)
    else:
        det_path= opt.eval
    import pdb;pdb.set_trace()
    save_patch_results(det_path)


