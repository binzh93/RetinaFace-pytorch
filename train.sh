# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u train.py > RetinaFace_c1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 nohup python -u train.py > RetinaFace_t1.log 2>&1 &
