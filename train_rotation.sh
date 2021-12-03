CUDA_VISIBLE_DEVICES=0 python3 train_rotation.py \
--weights yolov5s.pt \
--img 640 \
--workers 4 \
--name hrsc2016_2021_1126 \
--cfg models/yolov5s_hrsc2016.yaml \
--data data/hrsc2016.yaml  \
--hyp data/hyp.scratch.yaml \
--batch-size 44 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py