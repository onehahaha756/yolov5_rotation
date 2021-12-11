CUDA_VISIBLE_DEVICES=1 OPENCV_IO_MAX_IMAGE_PIXELS=20000000000 \
python3 detect_big_rotationV2.py \
--weights runs/train/seaship_rotation_ship_filter2/weights/best.pt  \
--dataset data/seaship_rotation_filter.yaml \
--project runs/detect/seaship_rotation_filter \
--conf 0.0001 \
--iou 0.05 \
--imgsz 640 \
--overlap 200 \
--remote \
--nosave \
--eval runs/detect/seaship_rotation_filter/exp12/results.pkl 
#--eval  runs/detect/dotav1_ship/exp2/results.pkl \
#--nosave 
#--eval runs/dotav2/exp69/results.pkl \
#--source /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/images \
#--annot_dir /data/03_Datasets/CasiaDatasets/Ship/MixShipV3/test_seaship/labelTxt 
#--nosave \
#--source /data/03_Datasets/CasiaDatasets/ShipOrigin/JL101K_PMS03_20200222111429_200022158_101_0013_001_L1/PAN/
#--source /data/03_Datasets/CasiaDatasets/Ship/SeaShip/image/
#--source /data/03_Datasets/CasiaDatasets/SeaShipOrigin/train
