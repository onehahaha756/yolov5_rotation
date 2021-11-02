CUDA_VISIBLE_DEVICES=0 python3 train_rotation.py \
--weights yolov5s.pt \
--img 640 \
--workers 4 \
--cfg models/yolov5s_seaship.yaml \
--data data/seaship_rotation.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 44 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py