CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
--weights yolov5s.pt \
--img 640 \
--cfg models/yolov5s.yaml \
--data data/seaship.yaml \
--hyp data/hyp.scratch.yaml \
--batch-size 64 \
--epochs 100 