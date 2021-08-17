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

pre_anchors=  [[6,12,  12,6,  12,26],  # P3/8
  [42,19,  28,52,  73,33], # P4/16
  [38,89,  286,93,  121,317] ] # P5/32

stride=[8.,16.,32.]
anchor_t=4.0

na=len(pre_anchors[0])//2
nl=len(pre_anchors)
pre_anchors=torch.tensor(pre_anchors).view(nl,na,2)

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
    # global pre_anchors
    # global anchor_t
    # global stride
    # import pdb;pdb.set_trace()
    nt=len(targets)
    p=[torch.ones(imgsz//int(down_sample),imgsz//int(down_sample)) for down_sample in stride]
    tcls, tbox, indices, anch= [], [], [], []
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
        anchors = pre_anchors[i]/stride[i]
        gain[2:6] = torch.tensor(p[i].shape)[[1, 0, 1, 0]]  # xyxy gain
        # import pdb;pdb.set_trace()
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
def draw_matched_h_anchor(img,indices,anch):
    # import pdb;pdb.set_trace()
    #get different layer
    for nl in range(len(indices)):
        cur_indice=indices[nl]
        cur_anch=anch[nl]
        # import pdb;pdb.set_trace()
        #get every matched anchor
        #get anchor index
        grid_x,grid_y=cur_indice[3].view(-1,1),cur_indice[2].view(-1,1)
        match_gridxy=torch.cat((grid_x,grid_y),1).float()
        #cat anchor x,y center and anchor w,h
        match_anchor=stride[nl]*torch.cat((match_gridxy,cur_anch),1)

        match_anchor=match_anchor.numpy().astype(np.int32)
        # import pdb;pdb.set_trace()
        for i in range(len(match_anchor)):
            x,y,w,h=match_anchor[i]
            x1,y1,x2,y2=x-w//2,y-h//2,x+w//2,y+h//2
            # import pdb;pdb.set_trace()
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2,2)
def draw_target_regresion_box(img,indices,tbox):
    '''
    verify regresion target is true
    '''
    for nl in range(len(indices)):
        cur_indice=indices[nl]
        cur_tbox=tbox[nl]
        if len(cur_tbox)<1:
            continue 
        match_tbox=cur_tbox.clone()
        # import pdb;pdb.set_trace()
        #get every matched anchor
        #get anchor index
        grid_x,grid_y=cur_indice[3].view(-1,1),cur_indice[2].view(-1,1)
        match_gridxy=torch.cat((grid_x,grid_y),1).float()
        # x,y
        # import pdb;pdb.set_trace()
        match_tbox[:,0:2]=match_gridxy+cur_tbox[:,0:2]
        match_tbox[:,4]*=90.

        #cat anchor x,y center and anchor w,h
        match_tbox[:,0:4]=stride[nl]*match_tbox[:,0:4]

        match_tbox=match_tbox.numpy().astype(np.int32)
        # import pdb;pdb.set_trace()
        for i in range(len(match_tbox)):
            x,y,w,h,theta=match_tbox[i]
            rect=((x,y),(w,h),theta)
            # import pdb;pdb.set_trace()
            bbox=cv2.boxPoints(rect).reshape((-1,1,2)).astype(np.int32)
            put_text='pos: {:.1f} {:.1f} {:.1f} {:.1f}'.format(x,y,w,h)
            cv2.putText(img,put_text,(50,100+nl*50),1,cv2.FONT_HERSHEY_PLAIN,(255,255,0),1)
            cv2.putText(img,str(theta),(400+50*i,100+nl*50),1,cv2.FONT_HERSHEY_PLAIN,(255,255,0),1)
            # import pdb;pdb.set_trace()
            cv2.polylines(img,[bbox],True,(255,255,0),2,2)

def draw_targets(img,targets):
    '''
    draw targets box
    '''
    show_targets=targets.clone()
    show_targets=show_targets.cpu().numpy()
    imgsz=img.shape[0]
    for i in range(len(show_targets)):
        x,y,w,h,theta=show_targets[i][2:]
        x*=imgsz 
        y*=imgsz 
        w*=imgsz 
        h*=imgsz 
        theta*=90
        # import pdb;pdb.set_trace()

        rect=((x,y),(w,h),theta)
        
        put_text='pos: {:.1f} {:.1f} {:.1f} {:.1f} {:.2f}'.format(x,y,w,h,theta)
        cv2.putText(img,put_text,(50,50),1,cv2.FONT_HERSHEY_PLAIN,(0,255,0),1)
        bbox=cv2.boxPoints(rect).reshape((-1,1,2)).astype(np.int32)

        cv2.polylines(img,[bbox],True,(0,255,0),2,2)

def vis_matched_anchor(targets,img):
    imgsz=img.shape[0]

    tcls, tbox, indices, anch= match_anchors(targets, imgsz)
    # import pdb;pdb.set_trace()
    draw_targets(img,targets)
    draw_matched_h_anchor(img,indices,anch)
    draw_target_regresion_box(img,indices,tbox)

    return img



def main(image_dir,label_dir,vis_dir):
    imglist=glob.glob(osp.join(image_dir,'*.jpg'))
    # import pdb;pdb.set_trace()
    for imgpath in tqdm(imglist):
            
        basename,suffix=osp.splitext(osp.basename(imgpath))
        annotpath=osp.join(label_dir,'{}.txt'.format(basename))
        vis_name=osp.join(vis_dir,'{}.jpg'.format(basename))


        img=cv2.imread(imgpath)
        
        targets=txt2targets(annotpath)

        #matched_anchors=match_anchors(targets,img.shape[0])
        show_img=vis_matched_anchor(targets,img)

        cv2.imwrite(vis_name,show_img)
        # import pdb;pdb.set_trace()
# main()
imgdir='/data/03_Datasets/dota_rotation/images/train'
annot_dir='/data/03_Datasets/dota_rotation/labels/train'

vis_dir='/data/03_Datasets/dota_rotation/vis_anchors/train'

if not osp.exists(vis_dir):
    os.makedirs(vis_dir)
main(imgdir,annot_dir,vis_dir)
  
    




    



