python3 utils/eval_casia.py \
--annot_dir /data/03_Datasets/CasiaDatasets/Ship/CutyoloShip/ori_labels/val \
--image_dir /data/03_Datasets/CasiaDatasets/Ship/CutyoloShip/ori_images/val \
--annot_type polygon \
--det_path runs/detect/exp6/results.pkl \
--clss ship \
--iou_thre 0.5 \
--conf_thre 0.3 \
--nms_thre 0.5