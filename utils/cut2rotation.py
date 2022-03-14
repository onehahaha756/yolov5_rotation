#-*- coding: utf-8 -*-
import os
import os.path as osp
import glob
import codecs,shutil
import random

import  xml.dom.minidom
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

#from dota2rotation import *
multil_type=['*.jpg','*.png','*.tif']
dataset_annotname=['ship']
dataset_annotname=['A','B','C','D','E','F','G','H','I','J','K']

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



def txt2polygons(txt_path):
    '''
    '''
    polygon_list = list()
    txt_file=codecs.open(txt_path,encoding='utf-8')

    for line in txt_file:
        #import pdb;pdb.set_trace()
        line=line.strip()
        row_list=line.split()
        polygon=[int(float(x)) for x in row_list[:8]]
        annot_name=row_list[-2]

        polygon_list.append((annot_name,polygon))

    txt_file.close()
    return polygon_list
def polygons2rotation(polygon_list):
    rotation_list=[]
    for polygon in polygon_list:
        class_name,points = polygon
        rect_array = np.array(points).reshape((-1, 2))
        rbbox = cv2.minAreaRect(rect_array)
        x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
        #some case angle is not in (0,90]
        while not 0 < a <= 90:
            if a <= 0:
                a += 90
                w, h = h, w
            else:
                a -= 90
                w, h = h, w
        rotation_list.append((class_name,(x,y,w,h,a)))
        
    return rotation_list

def rotation2scale(rotation_list,imgsize=640,theta_max=90):
    scale_rotation_list=[]
    for rotation in rotation_list:
        class_name,rotation_parm=rotation 
        # import pdb;pdb.set_trace()
        (x,y),(w,h),theta=rotation_parm
        
        x/=imgsize
        y/=imgsize

        w/=imgsize
        h/=imgsize

        theta/=theta_max
        # import pdb;pdb.set_trace()

        scale_rotation_list.append((class_name,(x,y,w,h,theta)))
    return scale_rotation_list


def scalepolygons2txt(save_path,scale_rotation_list):
    save_file = open(save_path, 'w', encoding='utf-8')
    for scale_rotation in scale_rotation_list:
        class_name,rotation_parm=scale_rotation 

        clss_id=dataset_annotname.index(class_name)

        x,y,w,h,theta=rotation_parm

        save_file.write('{} {} {} {} {} {}\n'.format(clss_id,x,y,w,h,theta))

    save_file.close()
def vis_rotation_labels(img,scale_rotation_list,visname,theta_max=90):
    show_img=img.copy()
    H,W,C=img.shape
    for scale_rotaion in scale_rotation_list:
        # import pdb;pdb.set_trace()
        annot_name=scale_rotaion[0]
        xc,yc,w,h,theta=scale_rotaion[1]
        #import pdb;pdb.set_trace()
        xc*=W
        yc*=H
        w*=W
        h*=H
        theta*=theta_max
        # import pdb;pdb.set_trace()
        rect=((xc,yc),(w,h),theta)
        box = cv2.boxPoints(rect).reshape(-1,1,2)
        box = np.int0(box)
        cv2.polylines(show_img,[box],isClosed=True,color=(0,255,0),thickness=2)
    #import pdb;pdb.set_trace()
    cv2.imwrite(visname, show_img)
def get_grid_list(img, roi_size=(400, 400), overlap_size=(50, 50)):
    ''' Calculate the bounding edges of cropping grids
    return:' xmin xmax ymin ymax'
    '''
    img_h, img_w = img.shape[0:2]
    #import pdb;pdb.set_trace()
    row_crops = (img_h - overlap_size[1]) // (roi_size[1] - overlap_size[1])
    col_crops = (img_w - overlap_size[0]) // (roi_size[0] - overlap_size[0])

    grid_list = [] # ymin, ymax, xmin, xmax
    for i_iter in range(row_crops * col_crops):

        x_crop_idx = i_iter % col_crops   #行
        y_crop_idx = i_iter//col_crops

        xmin=x_crop_idx*(roi_size[0]-overlap_size[0])
        ymin=y_crop_idx*(roi_size[1]-overlap_size[1])
        xmax=xmin+roi_size[0]
        ymax=ymin+roi_size[1]

        grid_list.append((xmin,xmax,ymin,ymax))

    return grid_list

def polygons2rect(polygon_list):
    rect_list=[]
    for polygon in polygon_list:
        annot_name=polygon[0]
        rect=polygon[1:]
        rect_array=np.array(rect).reshape((-1,2))
       # import pdb;pdb.set_trace()
        xmin=rect_array[:,0].min()
        xmax=rect_array[:,0].max()
        ymin=rect_array[:,1].min()
        ymax=rect_array[:,1].max()

        rect_list.append((annot_name,(xmin,ymin,xmax,ymax)))

    return rect_list

def rect2txt(rect_list,out_path):
    out_file=open(out_path,'w',encoding='utf-8')
    for rect in rect_list:
        annot_name=rect[0]
        clss_id=dataset_annotname.index(annot_name)
        rect_area=rect[1] #xmin,ymin,xmax,ymax
        out_file.write('{} {} {} {} {}\n'.format(clss_id,rect_area[0],rect_area[1],rect_area[2],rect_area[3]))
    out_file.close()

def match_grid_polygon(cur_grid,polygon):
    '''
    if rect in grid then return True and the new rect in subpatch
    cur_grid: [xmin:xmax,ymin:ymax]
    rect :[xmin,ymin,xmax,ymax]
    return
    rect_in_grid : type bool
    new_rect : new rect position in subpatch
    '''
    annot_name=polygon[0]
    #import pdb;pdb.set_trace()
    x1,y1,x2,y2,x3,y3,x4,y4=polygon[1]

    rect=polygon[1:]
    rect_array=np.array(rect).reshape((-1,2))
    # import pdb;pdb.set_trace()
    xmin=rect_array[:,0].min()
    xmax=rect_array[:,0].max()
    ymin=rect_array[:,1].min()
    ymax=rect_array[:,1].max()

    grid_xmin,grid_xmax,grid_ymin,grid_ymax=cur_grid
    rect_in_grid=False
    new_polygon=None
    if xmax<grid_xmax and xmin>grid_xmin and ymax<grid_ymax and ymin >grid_ymin:
        new_polygon=[annot_name,[x1-grid_xmin,y1-grid_ymin,x2-grid_xmin,y2-grid_ymin,x3-grid_xmin,y3-grid_ymin,x4-grid_xmin,y4-grid_ymin]]
        rect_in_grid=True
    return rect_in_grid,new_polygon


def match_grid_polygonv2(cur_grid,polygon):
    '''
    if rect in grid then return True and the new rect in subpatch
    cur_grid: [xmin:xmax,ymin:ymax]
    rect :[xmin,ymin,xmax,ymax]
    return
    rect_in_grid : type bool
    new_rect : new rect position in subpatch
    '''
    annot_name=polygon[0]
    #import pdb;pdb.set_trace()
    x1,y1,x2,y2,x3,y3,x4,y4=polygon[1]

    rect=polygon[1:]
    rect_array=np.array(rect).reshape((-1,2))
    # import pdb;pdb.set_trace()
    xmin=rect_array[:,0].min()
    xmax=rect_array[:,0].max()
    ymin=rect_array[:,1].min()
    ymax=rect_array[:,1].max()
    xc=(xmin+xmax)/2
    yc=(ymin+ymax)/2

    grid_xmin,grid_xmax,grid_ymin,grid_ymax=cur_grid
    rect_in_grid=False
    new_polygon=None
    if xc<grid_xmax and xc>grid_xmin and yc<grid_ymax and yc >grid_ymin:
        new_polygon=[annot_name,[x1-grid_xmin,y1-grid_ymin,x2-grid_xmin,y2-grid_ymin,x3-grid_xmin,y3-grid_ymin,x4-grid_xmin,y4-grid_ymin]]
        rect_in_grid=True
    return rect_in_grid,new_polygon


def vis_labels(img,rect_list,visname):
    show_img=img.copy()
    for rect in rect_list:
        # import pdb;pdb.set_trace()
        annot_name=rect[0]
        x1,y1,x2,y2=rect[1]
        #import pdb;pdb.set_trace()
        w=abs(x2-x1)
        h=abs(y2-y1)
        cv2.rectangle(show_img,(x1,y1),(x2,y2),(0,0,255),5)
        w_h='{}*{}'.format(w,h)
        cv2.putText(show_img,w_h,(x1-2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    #import pdb;pdb.set_trace()
    cv2.imwrite(visname, show_img)

def rect2scale(rect,cur_shape):
    Height,Width=cur_shape
    annot_name,rect_points=rect
    x1,y1,x2,y2=rect_points

    xc=(x1+x2)/2
    yc=(y1+y2)/2

    width=abs(x2-x1)
    height=abs(y2-y1)
    # import pdb;pdb.set_trace()
    xc/=Width
    yc/=Height

    width/=Width
    height/=Height

    # import pdb;pdb.set_trace()

    return [annot_name,[xc,yc,width,height]]

def cutimage2detect(input_imgdir,input_annotdir,save_imgdir,save_annotdir,cut_size=512,overlap=50,save_nage_ratio=0.1,vis=True,show_origin=False):
    '''
    input_imgdir: images 
    input_annotdir: txt format annots
    save_dir:output dataset dir
            ./train.txt
            ./test.txt
            ./images
            ./labels
    '''
    # target_lable_dir=os.path.join(save_dir,'labelDota')

    target_image_dir=save_imgdir
    target_lable_dir=save_annotdir
    dirname=osp.dirname(osp.dirname(target_image_dir))
    # import pdb;pdb.set_trace()

    dirs=[target_image_dir,target_lable_dir]

    for sub_dir in dirs:
        if osp.exists(sub_dir):
            shutil.rmtree(sub_dir)
            os.makedirs(sub_dir)
        else:
            os.makedirs(sub_dir)
    if vis:
        vis_dir=osp.join(dirname,'vis_labels')
        if not osp.exists(vis_dir):
            os.mkdir(vis_dir)
        else:
            shutil.rmtree(sub_dir)
            os.mkdir(sub_dir)           
    # if not os.path.exists(target_labletrain_dir):
    #     os.makedirs(target_labletrain_dir)
    # if not os.path.exists(target_imagetrain_dir):
    #     os.makedirs(target_imagetrain_dir)

    # if not os.path.exists(target_imagetest_dir):
    #     os.makedirs(target_imagetest_dir)
    # if not os.path.exists(target_labletest_dir):
    #     os.makedirs(target_labletest_dir)
    #show origin labels
    imglist=[]
    for imgtype in multil_type:
        imglist+=glob.glob(osp.join(input_imgdir,imgtype))

    assert len(imglist) >0,'imgdir is empty please check your image directory path!'
    save_img_nums=0
    print(f'total ori images : {len(imglist)}')
    for imgpath in tqdm(imglist):
            
        basename,suffix=osp.splitext(osp.basename(imgpath))
        annotpath=osp.join(input_annotdir,'{}.txt'.format(basename))
        
        img=cv2.imread(imgpath)
        
        polygon_list=txt2polygons(annotpath)

        # rect_list=polygons2rect(polygon_list)

        grid_list=get_grid_list(img,(cut_size,cut_size),(overlap,overlap))

        #
        for i,cur_grid in enumerate(grid_list):
            exsit_object=False
            new_polygonlist=[]
            #match rect withe grid
            for polygon in polygon_list:
                rect_in_grid,new_polygon=match_grid_polygonv2(cur_grid,polygon)
                if rect_in_grid:
                    exsit_object=True
                    # scale_new_polygon=rect2scale(new_polygon,(cut_size,cut_size))
                    new_polygonlist.append(new_polygon)
                   #import pdb;pdb.set_trace()
            if exsit_object:
                # save_name=basename+'{}'.format(i)
                rotation_list=polygons2rotation(new_polygonlist)
                # import pdb;pdb.set_trace()
                scale_rotation_list= rotation2scale(rotation_list,imgsize=cut_size)

                save_labelname=osp.join(target_lable_dir,'{}_{}.txt'.format(basename,i))
                save_imgname=osp.join(target_image_dir,'{}_{}.jpg'.format(basename,i))
                save_img=img[cur_grid[2]:cur_grid[3],cur_grid[0]:cur_grid[1],:]
                
                
               
                #print('save image {}'.format(save_imgname))
                scalepolygons2txt(save_labelname,scale_rotation_list)
                
                cv2.imwrite(save_imgname,save_img)

                if vis:
                    vis_name=osp.join(vis_dir,'{}_{}.jpg'.format(basename,i))
                    vis_rotation_labels(save_img,scale_rotation_list,vis_name,theta_max=90)
                save_img_nums+=1
            else:
                #continue
                #print('save img ',i)
                if random.random()>save_nage_ratio:
                    continue
                save_imgname=osp.join(target_image_dir,'{}_{}.jpg'.format(basename,i))
                save_img=img[cur_grid[2]:cur_grid[3],cur_grid[0]:cur_grid[1],:]
                cv2.imwrite(save_imgname,save_img)
                save_img_nums+=1
    print(f'total images: {save_img_nums} !')
    #split_traintest(target_image_dir,save_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="cut remote images into small patches ")

    parser.add_argument('--input_imgdir',help="directory of input images")
    parser.add_argument('--input_annotdir',help="directory of the txt format annot")
    parser.add_argument('--saveimg_dir',help="directory of images output")
    parser.add_argument('--saveannot_dir',help="directory of annot output")
    parser.add_argument('--save_nage_ratio',type=float,help="save nagetive samples ratio of total nagetive samples")
    parser.add_argument('--vis_label',default=False,type=str2bool,help="show origin labels")
    parser.add_argument('--resplit',default=False,type=str2bool,help="do not cut the dataset ,but only resplit train and testdata")
    parser.add_argument('-c','--cut_size',type=int,default=512,help="cut patch sizes")
    parser.add_argument('-l','--overlap',type=int,default=50,help="overlap for cut")


    args = parser.parse_args()
    # if osp.exists(args.save_dir):
    #     shutil.rmtree(args.save_dir)
    # else:
    #only resplit the dataset,do not recut
    if args.resplit:
        target_image_dir=os.path.join(args.save_dir,'image')
        #split_traintest(target_image_dir,args.save_dir)
    else:
        cutimage2detect(args.input_imgdir,args.input_annotdir,args.saveimg_dir,args.saveannot_dir,args.cut_size,args.overlap,args.save_nage_ratio,args.vis_label)




