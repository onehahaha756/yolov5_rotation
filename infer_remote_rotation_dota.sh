CUDA_VISIBLE_DEVICES=1 python3 detect_patch.py \
--weights runs/train/ship_direction_sub_train_640_3004/weights/best.pt \
--dataset data/ship_direction.yaml \
--project runs/detect/ship_direction \
--conf 0.8 \
--iou 0.1 \
--imgsz 1000 \
--max 2000 \
--overlap 600 \
--remote \
#--nosave \
#--eval runs/detect/dotav2/dotav2_skewiou_loss_v2_1024_600_/20200924_iou0.5_conf0.001_maxnms5000_maxdet2000/exp/results.pkl
#--eval runs/detect/dotav2/yolov5s-dotav2-skewiou-loss-1024-600/exp3/results.pkl 
#--source /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/images \
#--annot_dir /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/labelTxt 
#--nosave \
#--source /data/03_Datasets/CasiaDatasets/ShipOrigin/JL101K_PMS03_20200222111429_200022158_101_0013_001_L1/PAN/
#--source /data/03_Datasets/CasiaDatasets/Ship/SeaShip/image/
#--source /data/03_Datasets/CasiaDatasets/SeaShipOrigin/train
