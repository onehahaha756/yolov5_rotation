python3 utils/eval_rotation.py \
--annot_dir /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/labelTxt \
--image_dir /data/03_Datasets/DOTA-v2.0/val/labelTxt \
--annot_type polygon \
--datafile data/seaship.yaml \
--det_path runs/detect/exp125/results.pkl \
--iou_thre 0.3 \
--conf_thre 0.01 \
--nms_thre 0.5