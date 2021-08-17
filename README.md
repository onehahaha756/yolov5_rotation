
## 基于yolov5的旋转框目标检测
课题需要，需要做旋转目标检测，因此尝试了基于Yolov5的旋转目标检测。
### 修改的地方
1. 数据表示方式
- 飞机检测或舰船检测，原始数据集格式为Dota标注格式:
    [x1,y1,x2,y2,x3,y3,x4,y4,cls]
- yolov5标签格式
  yolov5正矩形框表示格式为[cls,x,y,w,h]，当前修改的模型新增加了一个旋转角度[cls,x,y,w,h,theta]，具体数值来自于cv2.minArea()，根据旋转框的4个点坐标，由opencv生成。
- opencv版本为4.5.1,x,y,w,h,theta定义参考:
   https://blog.csdn.net/zzzhaowendao/article/details/117510645
- 对x,y,w,h,theta进行坐标归一化
   (x/imgsz,y/imgsz,w/imgsz,h/imgsz,theta/90)
2. Dataloader修改（dataloader_rotation.py)
   修改一些加载函数
3. 数据增广（支持的增广方式及处理方式)
- mosic增广：mosic增广时，由于图片平移拼接时，部分目标会部分出界，此处的处理方式是：若旋转框的中心在图片中，则保留目标，否则则丢弃目标。目标的旋转框尺寸不变。
- 翻转：label的w,h坐标互换，角度取余角
- hsv调整: 与正矩形框相同
- 平移: 相应的平移xc,yc
4. 模型修改
- 修改模型Detect模块，增加bbox theta的维度
- loss修改
  简单起见，[x,y,w,h]依旧采用iou loss回归，theta用smoothL1回归，需要进一步改进
5. Nms修改
   Nms在测试的时候进行重叠框过滤，旋转框iou计算，调用detectron2的函数
   detectron2 : https://github.com/facebookresearch/detectron2/tree/66d658de02a2579d9516a72d94e98a394e2f0ccf/detectron2
6. 评价代码修改
   制作数据集时，将4096图像切成了640尺寸大小，测试时采用自己编写的代码，之前写的是针对矩形框的，需要针对旋转框修改下nms和评价的代码（to do）
### 从头开始训练及测试流程（代码使用说明)
1. dota标注数据集格式转yolov5 rotation标注格式
- 脚本 utils/cut2rotation.sh ，修改数据集路径和滑窗切片大小及重叠面积即可
2. 数据集路径修改 
- 修改data文件夹下数据集定义
3. anchor聚类
- cd utils，修改数据集路径，然后python autoanchors.py即可
4. 模型anchor修改
- 将聚类的anchor更新到模型定义中 models/yolov5s_rotation.yaml
5. 训练
- 脚本 dis_train_rotation.sh，修改数据路径运行
6. 测试及评价
- 脚本 infer_remote_rotation.sh，修改数据路径及模型权重，会保存测试图像结果，但目前需要进一步修改旋转框的评价代码，评测结果暂不可用
### 训练标签可视化
数据集地址:https://www.rsaicp.com/portal/contestDetail?id=2&tab=rule
![image](https://user-images.githubusercontent.com/49705914/129001296-1397d0ba-75bb-4a4d-ac70-26abaf0f0bc5.png)

dotav2数据集

![image](https://user-images.githubusercontent.com/49705914/129298756-d16c4855-b508-4ca3-a2d4-21fc3105ff1c.png)

### 测试结果
- 舰船AP       0.674

![image](https://user-images.githubusercontent.com/49705914/128113280-3f72c644-9297-4885-bf63-780a2f230124.png)
- 飞机11类mAP  0.82左右

![image](https://user-images.githubusercontent.com/49705914/128284942-27fe2008-83eb-47c2-8754-204cef5e60ad.png)
### utils
1. 锚框匹配可视化 
- utils/vis_match_anchors.py 
淡蓝色的框为回归到的位置，红色为匹配到的正矩形框，绿色为groudthrth，文字部分即回归的坐标及角度

![image](https://user-images.githubusercontent.com/49705914/128320551-6e9ddbdd-70bf-4ab2-a0db-4c365853240d.png)
![image](https://user-images.githubusercontent.com/49705914/128320616-c4b394d4-45cf-4ed8-acd1-e043bf02b034.png)




  

  
