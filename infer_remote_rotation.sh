CUDA_VISIBLE_DEVICES=1 python3 detect_big_rotation.py \
--weights runs/train/dotav2_skewiou_loss13/weights/last.pt  \
--conf 0.5 \
--iou 0.5 \
--imgsz 640 \
--overlap 200 \
--remote \
--source /data/03_Datasets/dota_rotationV2/images/val \
--annot_dir /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/labelTxt 
#--nosave \
#--source /data/03_Datasets/CasiaDatasets/ShipOrigin/JL101K_PMS03_20200222111429_200022158_101_0013_001_L1/PAN/
#--source /data/03_Datasets/CasiaDatasets/Ship/SeaShip/image/
#--source /data/03_Datasets/CasiaDatasets/SeaShipOrigin/train
