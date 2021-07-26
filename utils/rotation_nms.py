import os
import cv2
import glob
import numpy as np

def caculate_iou_rotation(polygons1, polygons2, mask_size=4096):
    '''
    polygons: array.numpy [x1,y1,x2,y2,x3,y3,x4,y4,score]
    '''
    # import pdb;pdb.set_trace()
    polygons1=np.array(polygons1[:8]).reshape(-1,2)
    polygons2=np.array(polygons2[:8]).reshape(-1,2)
    #get bonding rect bbox
    x1_min,x1_max=polygons1[:,0].min(),polygons1[:,0].max()
    y1_min,y1_max=polygons1[:,1].min(),polygons1[:,1].max()

    x2_min,x2_max=polygons2[:,0].min(),polygons2[:,0].max()
    y2_min,y2_max=polygons2[:,1].min(),polygons2[:,1].max()

    #计算最小正矩形包围盒是否相交
    x_maxmin=max(x1_min,x2_min)
    x_minmax=min(x1_max,x2_max)

    y_maxmin=max(y1_min,y2_min)
    y_minmax=min(y1_max,y2_max)

    w=max(x_minmax-x_maxmin,0)
    h=max(y_minmax-y_maxmin,0)
    rect_inserption=w*h
    #如果最小矩形包围盒不相交，则旋转包围盒一定不相交，返回0
    if rect_inserption==0:
        # print('no inception')
        iou=0

    else:
        mask_xmin=min(x1_min,x2_min)
        mask_ymin=min(y1_min,y2_min)

        mask_xmax=min(x1_max,x2_max)
        mask_ymax=min(y1_max,y2_max)
        #取包围盒跨度最大的x或y作为mask
        mask_size=max(mask_xmax-mask_xmin,mask_ymax-mask_ymin)
        mask_size=int(mask_size)+100
        # import pdb;pdb.set_trace()
        mask1 = np.zeros((mask_size, mask_size))
        mask2 = np.zeros((mask_size, mask_size))

        shift_vector=np.array([[mask_xmin,mask_ymin]])
        shift_vector=shift_vector.repeat(4,axis=0)
        polygons1 = (polygons1-shift_vector).astype(np.int32)
        polygons2 = (polygons2-shift_vector).astype(np.int32)

        mask1 = cv2.fillPoly(mask1, [polygons1], 1)
        mask2 = cv2.fillPoly(mask2, [polygons2], 1)

        inserption = np.sum(mask1 * mask2)
        union = mask1.sum() + mask2.sum() - inserption

        iou = inserption / float(union)
    # print(iou)

    return iou

def rotation_nms(boxes,scores,iou_thre):
    '''
    input:

    '''
    nms_polygons=[]
    boxes=boxes.cpu().numpy()
    scores=scores.cpu().numpy()
    #sort
    idx=np.argsort(-scores)
    boxes=boxes[idx]
    scores=scores[idx]
    for (box,score) in zip(boxes,scores):
        # import pdb;pdb.set_trace()
        xc,yc,w,h,theta=box
        rect=((xc,yc),(w,h),theta)
        polygons=cv2.boxPoints(rect)
        # import pdb;pdb.set_trace()
        polygons=np.hstack((polygons.reshape(8),score))
        save_box=True
        for nms_polygon in nms_polygons:
            iou_pre=caculate_iou_rotation(polygons,nms_polygon)
            if iou_pre>iou_thre:
                save_box=False
        if save_box:
            nms_polygons.append(polygons)
    return np.array(nms_polygons).reshape(-1,9)
         

def nms(predictions,iou_thre,conf_thre):
    '''
    input:
    predictions:(list),[x1,y1,x2,y2,score,clss],shape[nums_bboxes,6]
    iou_thre: nms overlap threshold
    conf_thre: confidence score to filter
    output:
    nms_bboxes:(list),[x1,y1,x2,y2,score,clss],shape[nums_bboxes,6]
    '''
    # import pdb;pdb.set_trace()
    if len(predictions)<2:
        return np.array(predictions).tolist()
    predictions=np.array(predictions)
    predictions=predictions[predictions[:,-1].argsort()] #sort by classes
    classes_num=predictions[:,-1].max()
    #import pdb;pdb.set_trace()
    nms_bboxes=[]
    clss_bboxes=[]
    for clss in range(int(classes_num+1)): 
        # clss 0 : background 
        clss_predictions=predictions[predictions[...,-1]==clss]
        clss_predictions=clss_predictions[np.argsort(-clss_predictions[...,-2])]
        clss_bboxes=[]
        for i in range(len(clss_predictions)):
            predict=clss_predictions[i]
            bb=predict[:-2]
            score=predict[-2]

            if score<conf_thre:
                continue

            if len(clss_bboxes)==0:
                clss_bboxes.append(predict)
                clss_bboxes=np.array(clss_bboxes)
                continue
            #import pdb;pdb.set_trace()
            ixmin = np.maximum(clss_bboxes[:, 0], bb[0])
            iymin = np.maximum(clss_bboxes[:, 1], bb[1])
            ixmax = np.minimum(clss_bboxes[:, 2], bb[2])
            iymax = np.minimum(clss_bboxes[:, 3], bb[3])

            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (clss_bboxes[:, 2] - clss_bboxes[:, 0]) *
                       (clss_bboxes[:, 3] - clss_bboxes[:, 1]) - inters)

            overlaps = inters / uni
            # 求与之前最大的iou
            if (overlaps<iou_thre).all():
                clss_bboxes=np.vstack((clss_bboxes,predict))
            # else:
            #     print('nms bbox {} ,iou {}\n'.format(predict.tolist(),overlaps.max()))
        if len(nms_bboxes)==0:
            nms_bboxes=clss_bboxes.copy()
            #nms_bboxes=np.a
        else:
            nms_bboxes=np.vstack((nms_bboxes,clss_bboxes))
        nms_bboxes=np.array(nms_bboxes).tolist()
    return nms_bboxes


def softnms(bbox,score,iou_thres,score_thres):
    S=score.copy()
    D=[]
    while len(D)<bbox.shape[0]:
        S[D]=0
        m=np.argmax(S)
        if S[m]<score_thres:
            break
        M=bbox[m]
        D.append(m)
        for i in range(bbox.shape[0]):
            if i not in D and iou(M,bbox[i])>=iou_thres:
                S[i]=S[i]*(1-iou(M,bbox[i]))
    return D


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="post processing of output")
    parser.add_argument('in_dir',nargs='+',help="directories of txt files")
    parser.add_argument('output_dir',help="directory of output")
    parser.add_argument('--mask_dir',help='Directory of the mask images')
    args = parser.parse_args()
    main(args)
