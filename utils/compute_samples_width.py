#coding:utf-8
import os,glob
import cv2 
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
subset_point=[0,10,20,30,40,50,60,70,80,90,100,110,150,200]

def txt2rect(txt_path):
    '''
    casia ship v1 annot format:[[x1,y1,x2,y2,x3,y3,x4,y4],[....]]
    '''
    rect_list = list()
    txt_file=open(txt_path,encoding='utf-8')

    for line in txt_file:
        #import pdb;pdb.set_trace()
        line=line.strip()
        row_list=line.split()
        annot_name=row_list[-2]
        polygon=[float(x) for x in row_list[:-2]]
        rect_list.append((annot_name,polygon))

    txt_file.close()
    return rect_list
def get_wh(rect_list):
    wh_list=[]
    for rect in rect_list:
        annot_name=rect[0]
        # import pdb;pdb.set_trace()
        x1,y1,x2,y2,x3,y3,x4,y4=rect[1]
        xmax=max(x1,x2,x3,x4)
        xmin=min(x1,x2,x3,x4)

        ymax=max(y1,y2,y3,y4)
        ymin=min(y1,y2,y3,y4)

        w=abs(xmax-xmin)
        h=abs(ymax-ymin)

        w,h = max(w,h),min(w,h)
        wh_list.append((w,h))
    return wh_list
def get_index(num):
    if num >= subset_point[-1]:
        return -1
    for i in range(len(subset_point)):
        if num<subset_point[i]:
            return i
def get_ws_hs(annot_dir):
    annot_list=glob.glob(osp.join(annot_dir,'*.txt'))
    ws=[0]*(len(subset_point)+1)
    hs=[0]*(len(subset_point)+1)
    WH=[]
    for annot_path in annot_list:
        rect_list=txt2rect(annot_path)
        wh_list=get_wh(rect_list)
        WH+=wh_list
        for wh in wh_list:
            w,h=wh
            w_subindex=get_index(w)
            h_subindex=get_index(h)
            # if w_subindex==-1 or h_subindex==-1:
            #     # import pdb;pdb.set_trace()
            ws[w_subindex]+=1
            hs[h_subindex]+=1
    import pdb;pdb.set_trace()
    print(f'total objects :{np.array(ws).sum()}')
    for i in range(1,len(subset_point)):
        print(f'{subset_point[i-1]}<w<{subset_point[i]} : {ws[i]}')
    print(f'w>{subset_point[-1]}: {ws[-1]}')
    print(f'total objects :{np.array(hs).sum()}')
    for i in range(1,len(subset_point)):
            print(f'{subset_point[i-1]}<h<{subset_point[i]} : {hs[i]}')
    print(f'h>{subset_point[-1]}: {hs[-1]}')

annot_dir='/data/03_Datasets/CasiaDatasets/Ship/SeaShip/labelDota/'
get_ws_hs(annot_dir)



