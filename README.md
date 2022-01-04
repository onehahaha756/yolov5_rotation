
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
3. 数据增广（支持的增广方式)
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
   参考latest分支detect_big_rotationV2.py，实现了大图预测以及评估大图的精度
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
- 脚本 infer_remote_rotation.sh，修改数据路径及模型权重，会保存测试图像结果
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

- dota数据集

![image](https://user-images.githubusercontent.com/49705914/134128980-6df7dfcf-a743-4e40-8195-1af6971c4ee8.png)

![image](https://user-images.githubusercontent.com/49705914/134129967-9fedab65-5b71-4271-ade6-ab86dd14a910.png)



### utils
1. 锚框匹配可视化 
- utils/vis_match_anchors.py 
淡蓝色的框为回归到的位置，红色为匹配到的正矩形框，绿色为groudthrth，文字部分即回归的坐标及角度

![image](https://user-images.githubusercontent.com/49705914/128320551-6e9ddbdd-70bf-4ab2-a0db-4c365853240d.png)
![image](https://user-images.githubusercontent.com/49705914/128320616-c4b394d4-45cf-4ed8-acd1-e043bf02b034.png)


###
分支说明：
- master分支: 修改如上所述，模型方面修改了Detect模块，角度回归方式采用smoothl1
- dcn-yolov5-rotation:引入DCN，尝试适应多尺度问题，开发中
- develop:修改loss函数，参考：scrdet
- latest : 做了一些改进，其中重要的部分包括 \
  1）添加了DOTA_devkit（需要参考工具安装说明安装) \
  2）运行infer_remote_rotation.sh 脚本，可以直接在大图上预测，并输出结果，且可以评估模型检测精度 \
### 代码说明
该代码是由于自己的课题需要修改的，主要用于学习交流。同时我也还在学习和开发中，因此代码可能并没有很好的整理，如果大家对这份代码感兴趣或者有使用问题欢迎大家跟我交流~

### 代码目前常见的问题说明
- dota转yolov5 rotation 代码说明 

   cut2rotation.py 代码的旋转框定义方式，x轴顺时针旋转，默认第一条边为w，另外一条边为h。代码转换在图片尺寸长宽不一致的情况下，会转换出错。可参考latest分支中utils/hrsc2016_2_rotation.py进行新的标签转换方式，另外相应的dataset_rotation.py，utils/plots.py等也需要修改。latest分支中已经改过了。
   关于opencv定义旋转框的方式，x轴顺时针旋转第一条边为w，另一条边为h，角度为（0，90]。部分版本中，逆时针旋转第一条边为h,另一条边为w，角度为[-90，0），其实定义方式是一致的，区别在于差90°。在代码里做相应的调整即可。
- 模型性能

  做了dota数据集的实验，FPN默认3个stage,即下采样（8，16，32），未使用多尺度训练和测试，单用训练集训练，最终结果dota1 mAP约60,性能不是很好，以下可供参考：
  yolov5l_rotation：\
  mAP: 0.6008265883151498\
  ap of each class: plane:0.8764534796039967, baseball-diamond:0.5904322702703031, bridge:0.49038861352180907, ground-track-field:0.48863373753994466, small-   vehicle:0.7766348619928898, large-vehicle:0.7069625557468652, ship:0.8429322193652917, tennis-court:0.8672936947004829, basketball-court:0.7419463367516996, storage-tank:0.8493569084547188, soccer-ball-field:0.35651165726395767, roundabout:0.53453295511373, harbor:0.6331742403609525, swimming-pool:0.03336907026438091, helicopter:0.22377622377622378\
  在hrsc2016上用训练集和验证集训练，测试集测试，AP约89.3，性能尚可
- 其他

  针对遥感领域，一阶段模型需要改进的还有很多，此代码仅做了一点表层的改动。本人又一直比较忙于其他事情，未来得及好好整理和改进代码，因此可能会有bug。看到有人关注我的代码还是很开心的，有时间的时候，我会好好整理一下.





  

  
