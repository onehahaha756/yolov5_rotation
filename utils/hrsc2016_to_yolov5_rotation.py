import os
import numpy as np
import glob
import cv2
import os.path as osp
import math
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import tqdm


def Xml2ScalePolygons(xml_path):
    polygon_list = []
    tree = ET.parse(xml_path)
    #import pdb;pdb.set_trace()
    width = int(tree.find('Img_SizeWidth').text)
    height = int(tree.find('Img_SizeHeight').text)
    HRSC_Objects = tree.find('HRSC_Objects')
    # import pdb;pdb.set_trace()
    for objects in HRSC_Objects.findall('HRSC_Object'):
        # import pdb;pdb.set_trace()
        clss_id = objects.find('Class_ID').text
        # bndbox=objects.find('bndbox')
        xc = float(objects.find('mbox_cx').text)/width
        yc = float(objects.find('mbox_cy').text)/height
        w = float(objects.find('mbox_w').text)/max(width,height)
        h = float(objects.find('mbox_h').text)/max(width,height)
        angle = 180*float(objects.find('mbox_ang').text)/math.pi
        polygon_list.append([clss_id, xc, yc, w, h, angle])

    return polygon_list


def LongsidePolygon2OpencvRotaion(polygon_list):
    for polygon in polygon_list:
        #clss_id, xc, yc, w, h,angle = polygon
        assert -90 <= polygon[5] <= 90, 'angle is not in[-90,90]'
        if polygon[5] <= 0:
            # import pdb;pdb.set_trace()
            polygon[5] += 90
            polygon[3], polygon[4] = polygon[4], polygon[3]
        # scale to (0,1]
        polygon[5] /= 90
    return polygon_list


def vis_rotation_labels(img, scale_rotation_list, visname, theta_max=90):
    show_img = img.copy()
    H, W, C = img.shape
    for scale_rotaion in scale_rotation_list:
        # import pdb;pdb.set_trace()
        #annot_name = scale_rotaion[0]
        annot_name, xc, yc, w, h, theta = scale_rotaion
        #import pdb;pdb.set_trace()
        xc *= W
        yc *= H
        w *= max(W,H)
        h *= max(W,H)
        theta *= theta_max
        # import pdb;pdb.set_trace()
        rect = ((xc, yc), (w, h), theta)
        box = cv2.boxPoints(rect).reshape(-1, 1, 2)
        box = np.int0(box)
        cv2.polylines(show_img, [box], isClosed=True,
                      color=(0, 255, 0), thickness=2)
    cv2.imwrite(visname, show_img)


def SaveAnnotTxt(polygon_list, save_annot_path):
    f = open(save_annot_path, 'w')
    for polygon in polygon_list:
        clss_id, xc, yc, w, h, angle = polygon
        f.write('{} {} {} {} {} {} \n'.format('0', xc, yc, w, h, angle))
    f.close()

#dataset_spit = 'Train'
imgdir = '/data/03_Datasets/HRSC2016/HRSC2016/Test/AllImages'
annot_dir = '/data/03_Datasets/HRSC2016/HRSC2016/Test/Annotations'

save_imgdir = "/data/03_Datasets/hrsc2016/images/train"
save_annotdir = '/data/03_Datasets/hrsc2016/labels/train'
vis_labeldir = '/data/03_Datasets/hrsc2016/vis_dir/train'


dirs = [save_imgdir, save_annotdir, vis_labeldir]
for dd in dirs:
    if not osp.exists(dd):
        os.makedirs(dd)
ori_imgpaths = glob.glob(osp.join(imgdir, "*.bmp"))
total_instance_num = 0
for imgpath in tqdm.tqdm(ori_imgpaths):
    basename = osp.splitext(osp.basename(imgpath))[0]
    annot_path = osp.join(annot_dir, '{}.xml'.format(basename))

    img = cv2.imread(imgpath)

    polygonlist = Xml2ScalePolygons(annot_path)
    polygonlist = LongsidePolygon2OpencvRotaion(polygonlist)
    total_instance_num += len(polygonlist)

    save_imgpath = osp.join(save_imgdir, '{}.jpg'.format(basename))
    save_annotpath = osp.join(save_annotdir, '{}.txt'.format(basename))
    vis_labelpath = osp.join(vis_labeldir, '{}.jpg'.format(basename))

    cv2.imwrite(save_imgpath, img)
    SaveAnnotTxt(polygonlist, save_annotpath)
    vis_rotation_labels(img, polygonlist, vis_labelpath)
print('total ship nums:',total_instance_num)