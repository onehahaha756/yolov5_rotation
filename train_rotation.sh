CUDA_VISIBLE_DEVICES=0,1 python3 train_rotation.py \
--weights yolov5s.pt \
--img 640 \
--workers 16 \
--cfg models/yolov5s.yaml \
--data data/seaship_rotation.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 64 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py