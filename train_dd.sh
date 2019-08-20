# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u train.py > RetinaFace_c1.log 2>&1 &
# /home/dc2-user/zhubin/wider_face/train
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u train_dd.py > RetinaFace_dd_t1.log 2>&1 &

