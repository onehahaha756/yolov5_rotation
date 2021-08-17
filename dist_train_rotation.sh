CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py \
--weights yolov5s.pt \
--img 640 \
--workers 16 \
--name seaship \
--cfg models/yolov5s_seaship.yaml \
--data data/seaship_rotation.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 84 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py