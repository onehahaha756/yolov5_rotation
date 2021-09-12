CUDA_VISIBLE_DEVICES=1 python3 detect_big_rotation.py \
--weights runs/train/SeashipsV2/yolov5s_seaship_augment_l1_loss-v2-640-300-ratio0_15-/weights/best.pt  \
--dataset data/seaship_rotation.yaml \
--project runs/detect/rotation_ship \
--conf 0.01 \
--iou 0.5 \
--imgsz 4096 \
--overlap 0 \
--remote \
--nosave \
#--eval runs/dotav2/exp69/results.pkl \
#--source /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/images \
#--annot_dir /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/labelTxt 
#--nosave \
#--source /data/03_Datasets/CasiaDatasets/ShipOrigin/JL101K_PMS03_20200222111429_200022158_101_0013_001_L1/PAN/
#--source /data/03_Datasets/CasiaDatasets/Ship/SeaShip/image/
#--source /data/03_Datasets/CasiaDatasets/SeaShipOrigin/train
