CUDA_VISIBLE_DEVICES=1 python3 detect_patch_clss.py \
--weights runs/train/seaship_clss_20211104113/weights/best.pt \
--dataset data/seaship_rotation.yaml \
--project runs/detect/seaship_rotation_clss \
--conf 0.5 \
--iou 0.1 \
--imgsz 640 \
--max 1000 \
--overlap 0 \
--remote \
--nosave \
#--eval runs/detect/dotav2/dotav2_skewiou_loss_v2_1024_600_/20200924_iou0.5_conf0.001_maxnms5000_maxdet2000/exp/results.pkl
#--eval runs/detect/dotav2/yolov5s-dotav2-skewiou-loss-1024-600/exp3/results.pkl 
#--source /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/images \
#--annot_dir /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/labelTxt 
#--nosave \
#--source /data/03_Datasets/CasiaDatasets/ShipOrigin/JL101K_PMS03_20200222111429_200022158_101_0013_001_L1/PAN/
#--source /data/03_Datasets/CasiaDatasets/Ship/SeaShip/image/
#--source /data/03_Datasets/CasiaDatasets/SeaShipOrigin/train
