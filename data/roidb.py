import os
import os.path as osp
import cv2
import numpy as np
# import copy 

from easydict import EasyDict as edict
__C = edict()
cfg = __C
cfg.SCALES = [(640, 640)]
cfg.FACE_LANDMARK = True
cfg.MIN_FACE = 0
cfg.COLOR_JITTERING = 0.125
cfg.USE_FLIPED = True

# /home/shanma/Workspace/zhubin/RetinaFace/data/retinaface/images/0--Parade/0_Parade_marchingband_1_849.jpg


def get_roidb(image_info, data_path):
    roidb = []
    
    for k, v in image_info.items():
        image_path = osp.join(data_path, k)
        # print(image_path)
        image = cv2.imread(image_path) # (H, W, C)

        boxes = []
        landmarks = []
        gt_classes = []
        blur = []

        for line in v:
            values = [float(val) for val in line.split()]
            x1, y1, w, h = values[: 4]
            
            # filter by bounding box
            if x1<0 or y1<0 or w<0 or h<0:
                continue
            if w<=cfg.MIN_FACE or h<=cfg.MIN_FACE:
                continue

            box = [x1, y1, x1+w, y1+w]
            landmark = values[4: 19]
            blur_val = values[-1]
            gt_class = 1

            boxes.append(box)
            landmarks.append(landmark)
            blur.append(blur_val)
            gt_classes.append(gt_class)
        boxes = np.array(boxes, dtype=np.float32)
        landmarks = np.array(landmarks, dtype=np.float32).reshape(-1, 5, 3)
        blur = np.array(blur, dtype=np.float32)
        gt_classes = np.array(gt_classes, dtype=np.int)

        roi = {
            'image_path': image_path, 
            'height': image.shape[0],
            'width': image.shape[1],
            'boxes': boxes,   # x1, y1, x2, y2
            'gt_classes': gt_classes,
            'blur': blur,
            'flipped': False
        }
        if cfg.FACE_LANDMARK:
            roi['landmarks'] = landmarks
        if len(boxes)>0:
            roidb.append(roi)
            if cfg.USE_FLIPED:
                roidb.append(get_flipped_roi(roi))
        else:
            print(roi['image_path'])
    # print("roidb: ", len(roidb))
    return roidb


def get_flipped_roi(roi):
    boxes = roi['boxes'].copy()
    oldx1 = boxes[:, 0].copy()
    oldx2 = boxes[:, 2].copy()
    boxes[:, 0] = roi['width'] - oldx2 - 1
    boxes[:, 2] = roi['width'] - oldx1 - 1
    assert (boxes[:, 2] >= boxes[:, 0]).all()

    flipped_roi = {
        'image_path': roi['image_path'], 
        'height': roi['height'],
        'width': roi['width'],
        'boxes': boxes,  
        'gt_classes': roi['gt_classes'].copy(),
        'blur': roi['blur'].copy(),  # TODO
        'flipped': True
    }

    if "image_data" in roi:
        image = roi['image_data'].copy()
        image = image[:, ::-1]
        flipped_roi['image_data'] = image

    if cfg.FACE_LANDMARK:
        landmarks = roi['landmarks'].copy()
        oldx_all = landmarks[:, :, 0]
        landmarks[:, :, 0] = roi['width'] - oldx_all - 1
        order = [1, 0, 2, 4, 3]
        landmarks_flipped = landmarks.copy()
        for k, v in enumerate(order):
            landmarks_flipped[:, k, :] = landmarks[:, v, :]
        flipped_roi["landmarks"] = landmarks_flipped

    return flipped_roi




