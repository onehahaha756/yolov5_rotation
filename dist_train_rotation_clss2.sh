CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation_clss.py \
--weights runs/train/seaship_clss_20211104116/weights/best.pt \
--img 640 \
--workers 16 \
--name seaship_clss_20211104 \
--cfg models/yolov5s_clss.yaml \
--data data/seaship_rotation.yaml   \
--hyp data/hyp.scratch.yaml \
--batch-size 88 \
--epochs 300 

# python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py