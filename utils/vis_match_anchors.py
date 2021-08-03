#coding:utf-8

import cv2 
import numpy as np
import torch
import os,glob
import os.path as osp 
import codecs
from tqdm import tqdm

pre_anchors=[[23,28, 68,57, 46,113],
    [143,44, 135,78, 98,127],
    [236,112, 148,209, 243,282]]

pre_anchors=[[30,10,  19,66,  71,23],  # P3/8
   [125,30,  30,135,  212,39],  # P4/16
  [45,248,  321,72,  69,351] ] # P5/32
def txt2targets(txt_path):
    '''
    '''
    txt_file=codecs.open(txt_path,encoding='utf-8')
    imgid=[0]
    targets=[]
    for line in txt_file:
        #import pdb;pdb.set_trace()
        line=line.strip()
        row_list=line.split()
        target=imgid+[float(x) for x in row_list[:]]
        targets.append(target)
    txt_file.close()
    targets=torch.tensor(targets)
    # import pdb;pdb.set_trace()
    return targets

def match_anchors(targets,imgsz):
    #torch.ones(imgsz*i/32,imgsz*i/32)
    global pre_anchors
    anchor_t=4.0
    import pdb;pdb.set_trace()
    nt=len(targets)
    p=[torch.ones(imgsz*i//32,imgsz*i//32) for i in [1,2,4]]
    na=len(pre_anchors[0])//2
    nl=len(pre_anchors)
    pre_anchors=torch.tensor(pre_anchors).view(nl,na,2)
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
    #import pdb;pdb.set_trace()
    for i in range(nl):
        anchors = pre_anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[1, 0, 1, 0]]  # xyxy gain
        import pdb;pdb.set_trace()
        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < anchor_t  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        g_theta=t[:,6].unsqueeze(1)
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 7].long()  # anchor indices
        # import pdb;pdb.set_trace()
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh,g_theta), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
    # import pdb;pdb.set_trace()
    return tcls, tbox, indices, anch


def main(image_dir,label_dir,vis_dir):
    imglist=glob.glob(osp.join(image_dir,'*.jpg'))
    # import pdb;pdb.set_trace()
    for imgpath in tqdm(imglist):
            
        basename,suffix=osp.splitext(osp.basename(imgpath))
        annotpath=osp.join(label_dir,'{}.txt'.format(basename))
        vis_name=osp.join(vis_dir,'{}.jpg'.format(basename))


        img=cv2.imread(imgpath)
        
        targets=txt2targets(annotpath)

        matched_anchors=match_anchors(targets,img.shape[0])
        import pdb;pdb.set_trace()
# main()
imgdir='/data/03_Datasets/CasiaDatasets/Ship/CutyoloMixShipV3_640_rotation/images/train/'
annot_dir='/data/03_Datasets/CasiaDatasets/Ship/CutyoloMixShipV3_640_rotation/labels/train/'

vis_dir='/data/03_Datasets/CasiaDatasets/Ship/CutyoloMixShipV3_640_rotation/vis_anchor_dir/train/'

if not osp.exists(vis_dir):
    os.makedirs(vis_dir)
main(imgdir,annot_dir,vis_dir)
  
    



