CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py \
--weights yolov5l.pt \
--img 640 \
--workers 16 \
--name tzplane \
--cfg models/yolov5l.yaml \
--data data/tzplane.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 24 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py