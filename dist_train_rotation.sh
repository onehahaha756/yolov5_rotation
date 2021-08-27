CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py \
--weights yolov5s.pt \
--img 640 \
--workers 16 \
--name dotav2_skewiou_loss \
--cfg models/yolov5s_rotation.yaml \
--data data/dotav2.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 64 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py