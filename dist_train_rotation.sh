CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation_dcn.py \
--weights yolov5l.pt \
--img 640 \
--workers 16 \
--name dotav1_ship_20211201_ \
--cfg models/yolov5l_dotav1_ship.yaml \
--data data/dotav1_ship.yaml     \
--hyp data/hyp.scratch.yaml \
--batch-size 24 \
--epochs 300 \
--resume 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py 