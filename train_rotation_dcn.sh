CUDA_VISIBLE_DEVICES=1 python3 train_rotation_dcn.py \
--weights none \
--img 640 \
--workers 0 \
--cfg models/yolov5s_rotation_dcn.yaml \
--data data/tzplane.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 16 \
--epochs 500 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py