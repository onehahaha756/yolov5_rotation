for weights in "dotav2_skewiou_loss13","dotav2_skewiou_loss-ll","dotav2_skewiou_loss-v2",\

CUDA_VISIBLE_DEVICES=0 python3 detect_big_rotation.py \
--weights runs/train/tzplanes/tzplane41/weights/best.pt  \
--dataset data/dotav2_val.yaml \
--project runs/detect/dota/yolov5s-dotav2-skewiou-loss-1024-600 \
--conf 0.001 \
--iou 0.5 \
--imgsz 4096 \
--overlap 600 \
--remote \
--nosave 