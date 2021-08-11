CUDA_VISIBLE_DEVICES=1 python3 detect_rotation.py \
--weights runs/train/tzplane27/weights/best.pt  \
--conf 0.01 \
--iou 0.5 \
--imgsz 4096 \
--overlap 0 \
--remote \
--save_json submit/aircraft_results_val.json \
--source  /data/03_Datasets/CasiaDatasets/Plane/TZ_Plane_Dotaformat/val/images \
--annot_dir /data/03_Datasets/CasiaDatasets/Plane/TZ_Plane_Dotaformat/val/labelTxt 
#--device cpu 

#--nosave \
#--source /data/03_Datasets/CasiaDatasets/ShipOrigin/JL101K_PMS03_20200222111429_200022158_101_0013_001_L1/PAN/
#--source /data/03_Datasets/CasiaDatasets/Ship/SeaShip/image/
#--source /data/03_Datasets/CasiaDatasets/SeaShipOrigin/train
