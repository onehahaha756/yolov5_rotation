CUDA_VISIBLE_DEVICES=1 python3 detect_big_rotation.py \
--weights runs/train/dotav2_skewiou_loss-v2_/weights/best.pt  \
--dataset data/dotav2.yaml \
--project runs/dotav2 \
--conf 0.01 \
--iou 0.5 \
--imgsz 4096 \
--overlap 1024 \
--remote \
#--eval runs/dotav2/exp56/results.pkl \
#--source /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/images \
#--annot_dir /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/labelTxt 
#--nosave \
#--source /data/03_Datasets/CasiaDatasets/ShipOrigin/JL101K_PMS03_20200222111429_200022158_101_0013_001_L1/PAN/
#--source /data/03_Datasets/CasiaDatasets/Ship/SeaShip/image/
#--source /data/03_Datasets/CasiaDatasets/SeaShipOrigin/train
