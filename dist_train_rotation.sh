CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py \
--weights yolov5s.pt \
--img 640 \
--workers 16 \
--name ship_direction_sub_train_640_300 \
--cfg models/yolov5s_seaship.yaml \
--data data/ship_direction.yaml     \
--hyp data/hyp.scratch.yaml \
--batch-size 88 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py