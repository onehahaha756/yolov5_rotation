CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node 1  train_rotation.py \
--weights yolov5s.pt \
--img 1024 \
--workers 16 \
--test_frequce 3 \
--name dotav2_1024_600/20210921_yolov5s_l1_loss- \
--cfg models/yolov5s_dotav2.yaml \
--data data/dotav2.yaml   \
--hyp data/hyp.scratch.yaml \
--batch-size 14 \
--epochs 300 

#python3 -m torch.distributed.launch --nproc_per_node 2  train_rotation.py