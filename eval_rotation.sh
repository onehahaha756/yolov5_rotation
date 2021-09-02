python3 utils/eval_rotation_anylysis.py \
--annot_dir /data/03_Datasets/DOTA-v2.0/val/labelTxt \
--image_dir /data/03_Datasets/DOTA-v2.0/val/images \
--annot_type polygon \
--datafile data/dotav2.yaml \
--det_path runs/dotav2/exp57/results.pkl \
--iou_thre 0.5 \
--conf_thre 0.01 \
--nms_thre 0.5