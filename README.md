#Yolov5 rotation modified
## 基于yolov5的旋转框目标检测
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
2. Dataloader修改
   关闭了数据增广（to do next），调整了dataloader加载的格式。
3. 模型修改
- 修改模型Detect模块，增加bbox theta的维度
- loss修改
  简单起见，[x,y,w,h]依旧采用iou loss回归，theta用smoothL1回归，需要进一步改进
4. Nms修改
   Nms在测试的时候进行重叠框过滤，旋转框iou计算，调用detectron2的函数
   detectron2 : https://github.com/facebookresearch/detectron2/tree/66d658de02a2579d9516a72d94e98a394e2f0ccf/detectron2
5. 评价代码修改
   制作数据集时，将4096图像切成了640尺寸大小，测试时采用自己编写的代码，之前写的是针对矩形框的，需要针对旋转框修改下nms和评价的代码（to do）
### 测试结果
- 舰船AP       0.674
- 飞机11类mAP  0.41左右
需要进一步排查原因，修改loss函数等



  

  