CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation_dcn.py \
--img 640 \
--weights none \
--workers 16 \
--name dotav2_dcn \
--cfg models/yolov5s_rotation_dcn.yaml \
--data data/dotav2.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 80 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py