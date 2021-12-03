CUDA_VISIBLE_DEVICES=0 python3 train_rotation_clss.py \
--weights yolov5s.pt \
--img 640 \
--workers 6 \
--name seaship_clss_20211104 \
--cfg models/yolov5s_clss.yaml \
--data data/seaship_rotation.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 32 \
--epochs 300 \
#--resume 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py