# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py
CUDA_VISIBLE_DEVICES=0,1 nohup python -u train.py > RetinaFace_new_res50.log 2>&1 &
