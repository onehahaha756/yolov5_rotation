#coding:utf-8
import numpy as np
import cv2
import math

def rboxes2points(pred,CLASSES,score_thr=0):
    # import pdb;pdb.set_trace()
    pred=np.array(pred)
    assert pred.shape[1] == 6 or pred.shape[1] == 7
    labels=pred[:,-1]
    scores=pred[:,-2]
    bboxes=pred[:,:5]
    results_list=[]
    if score_thr > 0:
        assert pred.shape[1] == 7
        inds = scores > score_thr
        bboxes = pred[inds, :]
        labels=labels[inds]
        scores=scores[inds]
    
    for label,bbox,score in zip(labels,bboxes,scores):
        object_dict={}
        xc, yc, w, h, ag = bbox.tolist()
        wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
        hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        #ps = np.int0(np.array([p1, p2, p3, p4]))
        label_text=CLASSES[int(label)]
        object_dict['category_id']=label_text
        object_dict['points']=[[float(x[0]),float(x[1])] for x in [p1,p2,p3,p4]]
        object_dict['confidence']=float(score)
        results_list.append(object_dict)
    return results_list

def save_jsonfile(imgname,save_results)