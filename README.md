
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
2. Dataloader修改（dataloader_rotation.py)
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
### 测试结果
- 舰船AP       0.674
![image](https://user-images.githubusercontent.com/49705914/128113280-3f72c644-9297-4885-bf63-780a2f230124.png)

- 飞机11类mAP  0.41左右
![image](https://user-images.githubusercontent.com/49705914/128113962-a6364e76-e90f-4ef4-b53e-70fded61dc90.png)
![image](https://user-images.githubusercontent.com/49705914/128113905-bf5171a4-bcd3-44e5-9269-ceee65f2284e.png)

准确率很高，但是有些类别召回率比较低，需要进一步排查
需要进一步排查原因，修改loss函数等



  

  
