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
from DOTA_devkit.dota_evaluation_task1 import voc_eval
from detectron2.layers import nms_rotated
multi_img_type=['*.jpg','*.png','*warp_uint8.tif','*.tiff']
merge_nms_iou_thr_dict = {
    'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.1,
    'soccer-ball-field': 0.3, 'small-vehicle': 0.05, 'ship': 0.05, 'plane': 0.3,
    'large-vehicle': 0.05, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
    'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3
}

import math

@torch.no_grad()
def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           clssname='',
           imgsz=640,  # inference size (pixels)
           overlap=200,# cut subimage overlap for remote images
           conf_thres=0.01,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           nosave=False,  # do not save images/videos
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           max_det=2000,  # maximum detections per image
           remote=True, #infer remote big
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
    if remote:
        imglist=[]
        for img_type in multi_img_type:
            imglist+=glob.glob(os.path.join(source,img_type))
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    vis_dir=osp.join(save_dir,'vis_results')
    if not osp.exists(vis_dir):
        os.mkdir(vis_dir)
    det_results={}
    save_results=[]
    # import pdb;pdb.set_trace()
    for i,imgpath in enumerate(imglist):
        ts=time.time()
        p = Path(imgpath)
        basename=os.path.splitext(os.path.basename(imgpath))[0]
        # import pdb;pdb.set_trace()
        try:
            ori_img=cv2.imread(imgpath)
            H,W,C=ori_img.shape
            #H_pad=check_img_size(H,s=stride)
            #W_pad=check_img_size(W,s=stride)
        except:
            print('read image error')
            continue
        # imgsz=min(imgsz,max(H_pad,W_pad))
        print('read time {}s\ndetecting...'.format(time.time()-ts))
        ts=time.time()
        step_h,step_w=(imgsz-overlap),(imgsz-overlap)
        ori_preds=torch.empty((0,7)).cuda()
        for x_shift in range(0,W,step_w):
            for y_shift in range(0,H,step_h):
                y_boader,x_boader=min(imgsz,H-y_shift),min(imgsz,W-x_shift)
                img=np.zeros((imgsz,imgsz,3))
                #print('{}/{}...'.format(x_shift//step_w,W//step_w))
                try:
                    img[:y_boader,:x_boader,:]=ori_img[y_shift:y_shift+y_boader,x_shift:x_shift+x_boader,:]
                except:
                    continue
                img=img.transpose(2,0,1)
                img = torch.from_numpy(img).to(device)

                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = model(img, augment=augment)[0]
                # Apply NMS
                # import pdb;pdb.set_trace()
                
                pred = non_max_suppression_rotation(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                #1 batch
                pred=pred[0]
                pred[:,0]+=x_shift
                pred[:,1]+=y_shift
                ori_preds=torch.cat((ori_preds,pred),0)
        print('detect time {}s'.format(time.time()-ts))
        if len(ori_preds)>0:
            # import pdb;pdb.set_trace()
            #nms by class shift
            before_nms=len(ori_preds)
            bbox_xy=ori_preds[:,:2]+imgsz*ori_preds[:,-1].reshape(-1,1)
            bboxes=torch.cat((bbox_xy,ori_preds[:,2:5]),1)
            scores=ori_preds[:,5]
            # import pdb;pdb.set_trace()
            nms_index=nms_rotated(bboxes,scores,iou_thres)
            # import pdb;pdb.set_trace()
            ori_preds=ori_preds[nms_index]
            end_nms=len(ori_preds)
            print(f'nms {before_nms-end_nms} bboxes')
            pred=[pd.cpu().numpy().tolist() for pd in ori_preds]

            p = Path(imgpath)
            print(f'detected {len(pred)} objects!')
            if save_json:
                img_result=rboxes2points(pred,clssname)
                save_result={}
                save_result['image_name']=osp.basename(imgpath)
                save_result['labels']=img_result
                save_results.append(save_result)            
            # import pdb;pdb.set_trace()
            if save_img:
                import random 
                if len(clssname) == 1:
                    colors = [(0,0,255)]
                else:
                    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(clssname))]
                show_img=ori_img.copy()
                show_img2=draw_clsdet_rotation(show_img,pred,clssname,colors,0.5) 
                # import pdb;pdb.set_trace()
                save_path = osp.join(vis_dir,'{}.jpg'.format(basename))  
                cv2.imwrite(save_path,show_img2)  
                print(f'{save_path} saved!')
            pred=[pd[:-1]+[clssname[int(pd[-1])]] for pd in pred]
            det_results[basename]=pred
        print(f'{i}/{len(imglist)}  processing {p.name}  shape:{(H,W,C)} ({time.time()-ts:.3f}s ETA: {(time.time()-ts)*(len(imglist)-i):.3f}s)')

    det_file=os.path.join(save_dir,'results.pkl')
    if save_json!=None:
        save_result_file=open(save_json,'w')
        json.dump(save_results,save_result_file,indent=2)
        save_result_file.close()
        print('save json file {} succeed!'.format(save_result_file))        
    with open(det_file, 'wb') as f:
        pickle.dump(det_results, f, pickle.HIGHEST_PROTOCOL)
        print(f'save {det_file} successed!')
    print(f'Done. ({time.time() - t0:.3f}s)')
    return det_file

def eval_remote(test_imagefile,annot_dir,annot_type,det_path,clssname,iou_thre,conf_thre,opt):
    '''
    use DOTA_devkit tools to eval results
    
    '''
    det_dir=os.path.dirname(det_path)
    results_path=osp.join(det_dir,'results.txt')
    
    save_file=open(results_path,'w',encoding='utf-8')
    save_file.write('imgsource: {}\nweights: {}\n'.format(test_images,opt.weights))
    save_file.write('iou overthre:{}\nConfidence thre:{}\n'.format(iou_thre,conf_thre))
    plt.figure(1)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('recall');plt.ylabel('presicion')
    map=0.0
    nc=len(classnames)
    


    # import pdb;pdb.set_trace()
    submit_path=osp.join(det_dir,'submit','Task1_{:s}.txt')
    annot_path=osp.join(annot_dir,'{:s}.txt')

    for clss in clssname:
        try:
            rec,prec,ap=voc_eval(submit_path,annot_path,test_imagefile, clss,ovthresh=0.3,use_07_metric=True)
            print('{:<20} : {:<20.5}  maxrecall: {:<20.5}  total pred: {:<20}'.format(clss,ap,rec[-1],len(rec)))
            save_file.write('{:<20} : {:<20.5}  total pred: {:<10}\n'.format(clss,ap,len(rec)))
            plt.plot(rec,prec,label=clss)
            save_fig_path=osp.join(det_dir,'{}_AP.png'.format(clss))
            plt.text(0.1,1.0,'nms iou thre:{} conf_thre: {} ap: {:.6f}'.format(iou_thre,conf_thre,ap))
            plt.savefig(save_fig_path)
            plt.clf()
            map+=ap/float(nc)
        except:
            save_file.write('{} erro \n'.format(clss))
            
            pass
    print('map : {:<20.5}'.format(map))
    save_file.write('map : {:<20.5}'.format(map))
    
    save_file.close()
   
   # plt.savefig(save_ap_fig)
    # with open(results_path,'w',encoding='utf-8') as f:
    #     f.write('imgsource: {}\nweights: {}\n'.format(opt.source,opt.weights))
    #     f.write('iou overthre:{}\nConfidence thre:{}\nAP:{}\nMaxRecall:{} \nMinPrecision: {}\n'\
    #             .format(iou_thre,conf_thre,ap,rec[-1],prec[-1]))
    # f.close()
def save_det2dota(det_path):
    dirname=osp.dirname(det_path)
    submit_dir=osp.join(dirname,'submit')
    if osp.exists(submit_dir):
        shutil.rmtree(submit_dir)
        os.mkdir(submit_dir)
    else:
        os.mkdir(submit_dir)
    det_file=open(det_path,'rb')
    det_dict=pickle.load(det_file)
    # import pdb;pdb.set_trace()
    for imgname in det_dict.keys():
        img_results=det_dict[imgname]
        for result in img_results:
            # import pdb;pdb.set_trace()
            xc, yc, w, h, ag,score,classname=result
            bbox_points=cv2.boxPoints(((xc,yc),(w,h),ag)).astype(np.int32)
            bbox_points=bbox_points.reshape(1,-1)[0].tolist()
            bbox_points=[max(0,x) for x in bbox_points]
            x1,y1,x2,y2,x3,y3,x4,y4=bbox_points
            
            save_path=osp.join(submit_dir,'Task1_{}.txt'.format(classname))
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
    parser.add_argument('--max-det', type=int, default=2000, help='maximum detections per image')
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
    #check_requirements(exclude=('tensorboard', 'thop'))
    import yaml 
    dataconfig=open(opt.dataset,'r',encoding='utf-8')
    dataset=yaml.load(dataconfig,Loader=yaml.FullLoader)
    classnames=dataset['names']

    test_images=dataset['test_images']
    test_labels=dataset['test_labels']
    test_imagefile=dataset['test_imgfile']
    clssname=dataset['names']
    # import pdb;pdb.set_trace()
    if not opt.eval:
        det_path=detect(opt.weights,test_images,clssname,opt.imgsz,opt.overlap,
                    opt.conf_thres,opt.iou_thres,opt.nosave,opt.project,opt.name)
    else:
        det_path= opt.eval
    save_det2dota(det_path)
    #imglist=glob.glob(os.path.join(opt.source,'*.jpg'))
    # import pdb;pdb.set_trace()
    eval_remote(test_imagefile,test_labels,'polygon',det_path,clssname,opt.iou_thres,opt.conf_thres,opt)
