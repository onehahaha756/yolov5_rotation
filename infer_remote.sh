CUDA_VISIBLE_DEVICES=1 python3 detect_remote.py \
--weights runs/train/exp200/weights/last.pt  \
--conf 0.7 \
--iou 0.05 \
--imgsz 640 \
--overlap 300 \
--remote \
--source /data/03_Datasets/CasiaDatasets/Ship/MixShipV2/train/images \
--annot_dir /data/03_Datasets/CasiaDatasets/Ship/MixShipV2/train/labelTxt

#--nosave \
#--source /data/03_Datasets/CasiaDatasets/ShipOrigin/JL101K_PMS03_20200222111429_200022158_101_0013_001_L1/PAN/
#--source /data/03_Datasets/CasiaDatasets/Ship/SeaShip/image/
#--source /data/03_Datasets/CasiaDatasets/SeaShipOrigin/train
