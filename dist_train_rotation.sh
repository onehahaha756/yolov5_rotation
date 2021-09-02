CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py \
--weights yolov5s.pt \
--img 1024 \
--workers 16 \
--name tzplane_skewiou_loss-v2-1024-600_ \
--cfg models/yolov5s_tzplane.yaml \
--data data/tzplane.yaml     \
--hyp data/hyp.scratch.yaml \
--batch-size 32 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py