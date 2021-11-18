CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation_clss.py \
--weights yolov5s.pt \
--img 1024 \
--workers 16 \
--name seaship_clss_1024_20211116_ \
--cfg models/yolov5s_clss.yaml \
--data data/seaship_rotation.yaml   \
--hyp data/hyp.scratch.yaml \
--batch-size 32 \
--epochs 300 

# python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py