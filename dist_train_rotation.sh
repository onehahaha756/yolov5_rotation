CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py \
--weights yolov5s.pt \
--img 640 \
--workers 16 \
--resume \
--name seaship_SkewiouLoss/yolov5s_seaship_augment_skewiou_loss-v2-640-300-0_15- \
--cfg models/yolov5s_seaship.yaml \
--data data/seaship_rotation.yaml     \
--hyp data/hyp.scratch.yaml \
--batch-size 88 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py