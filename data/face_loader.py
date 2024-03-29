#-*- coding: utf-8 -*-
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import cv2
import torch
from .data_augment import *
from .roidb import get_roidb
import time
import random


# landmark  -1: no landmark
# landmark  1.0: occlusion 
# landmark 0.0: normal
from easydict import EasyDict as edict
__C = edict()
cfg = __C

# cfg.Landmark = True
cfg.FACE_LANDMARK = False
cfg.MIN_FACE = 0
cfg.USE_FLIPED = True
cfg.COLOR_MODE = 2


# class FaceTransform(object):
#     def __init__(self):
#         pass

#     def __call__(self, target, landmark, width, height):
#         boxes = []
#         landmarks = []
#         labels = 1
#         return None



class WiderFaceDetection(data.Dataset):

    def __init__(self, root_path, data_path, phase="train", dataset_name="WiderFace", transform=None):
        self.name = dataset_name
        self.root_path = root_path
        # self.target_transform = FaceTransform()
        self.transform = transform

        self.data_path = osp.join(data_path, "images")
        txt_file = osp.join(self.root_path, phase, "label_ori.txt")
        # txt_file = osp.join(self.root_path, phase, "label.txt")

        self.image_info = {}
        self.load_info(txt_file)
        # print(self.image_info)

        self.roidb = get_roidb(self.image_info, self.data_path)
        # print(self.roidb)

        print("WiderFaceDetection:", len(self.roidb))
        # self.get_roidb()
        
    def __getitem__(self, index):
        image, boxes, landmarks = self.pull_item(index)
#         print("get_val")
#         print(image.shape)
#         print(boxes.shape)
#         print(landmarks.shape)
#         print("end")
#         print("img path: ", self.roidb[index]['image_path'])
        return image, boxes, landmarks, 
        
    def __len__(self):
#         print("__len__: ", len(self.roidb))
        return len(self.roidb)
    
    def load_info(self, txt_file):
        with open(txt_file, "r") as fr:
            for line in fr:
                if line.startswith("#"):
                    image_name = line.strip().split()[-1]
                    self.image_info[image_name] = []
                    continue
                self.image_info[image_name].append(line.strip())

    def pull_item(self, idx):
        
        roi = self.roidb[idx]
        t1 = time.time()
        DEBUG_I = False
        if DEBUG_I:
            img = cv2.imread(roi['image_path'])      
            bbxes = roi['boxes']
            lmks = roi['landmarks']#.reshape(-1, 15)
            print(roi['image_path'])
            if roi['flipped']:
                img = img[:, ::-1]
                print("images/z_flipped_" + osp.basename(roi['image_path']))
                cv2.imwrite("images/z_flipped_" + osp.basename(roi['image_path']), img)
                img = cv2.imread("images/z_flipped_" + osp.basename(roi['image_path']))
             
           
            for jj in range(bbxes.shape[0]):
                sf, st = (int(bbxes[jj][0]), int(bbxes[jj][1])), (int(bbxes[jj][2]), int(bbxes[jj][3]))
                print(sf, st)
                # print(lmks[jj])
                # print()
                cv2.rectangle(img, sf, st, (0, 0, 255), thickness=2)
                print((lmks[jj][0, 0],lmks[jj][0, 1]))
                print((lmks[jj][1, 0],lmks[jj][1, 1]))
                print((lmks[jj][2, 0],lmks[jj][2, 1]))
                print((lmks[jj][3, 0],lmks[jj][3, 1]))
                print((lmks[jj][4, 0],lmks[jj][4, 1]))
                cv2.circle(img,(lmks[jj][0, 0],lmks[jj][0, 1]),radius=1,color=(0,0,255),thickness=2)
                cv2.circle(img,(lmks[jj][1, 0],lmks[jj][1, 1]),radius=1,color=(0,255,0),thickness=2)
                cv2.circle(img,(lmks[jj][2, 0],lmks[jj][2, 1]),radius=1,color=(255,0,0),thickness=2)
                cv2.circle(img,(lmks[jj][3, 0],lmks[jj][3, 1]),radius=1,color=(0,255,255),thickness=2)
                cv2.circle(img,(lmks[jj][4, 0],lmks[jj][4, 1]),radius=1,color=(255,255,0),thickness=2)
            if roi['flipped']:
                cv2.imwrite("images/flipped_" + osp.basename(roi['image_path']), img)
            else:
                cv2.imwrite("images/" + osp.basename(roi['image_path']), img)

        roi = crop_img(roi)
        t2 = time.time()
        time_threshold = 1 # 0.03
        if t2 - t1 > time_threshold:
            print("crop time: ", t2 - t1)

        # if self.transform is not None:
        #     roi = self.transform(roi)
        if 'image_data' in roi:
            image =  roi['image_data']
        else:
            image = cv2.imread(roi['image_path'])
#         if roi['flipped']:
#             image = image[:, ::-1]
        isColor_JITTERING = True
        if isColor_JITTERING:
            image = image.astype(np.float32)
            image = color_aug(image, 0.125)
        
            
        # image = image.astype(np.float32) # TODO if must ????
        # PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
        # PIXEL_MEANS = np.array([0.0, 0.0, 0.0])
        # PIXEL_STDS = np.array([1.0, 1.0, 1.0])
        PIXEL_MEANS = np.array([0.406,0.456, 0.485])  # bgr mean
        PIXEL_STDS = np.array([0.225, 0.224, 0.229])
        PIXEL_SCALE = 255.0
        image = transform_ms(image, PIXEL_MEANS, PIXEL_STDS, PIXEL_SCALE) # already to NCHW
        image = image.astype(np.float32) # TODO if must ????
        # image = roi['image_data']
        # image = cv2.imread(roi['image_path'])
        # print("image_data is in roi: ", roi['image_data'].shape)
        # image = roi['image_data']

        # image = image[:, :, (2, 1, 0)]  # to rgb
        im_tensor = torch.from_numpy(image)#.permute(2, 0, 1)
        # print(roi['landmarks'])

        box_include_class = True
        if box_include_class:
            gt_classes = roi['gt_classes'].copy()
            boxes = roi['boxes'].copy()
            boxes = np.hstack((boxes, gt_classes[:, np.newaxis]))
            return im_tensor, boxes, roi['landmarks']
        else:
            return im_tensor, roi['boxes'], roi['landmarks']

  # if cfg.COLOR_JITTERING>0.0:
    #     pass
        # im = im.astype(np.float32)
        # im = color_aug(im, config.COLOR_JITTERING)
    # im_tensor = transform(im, config.PIXEL_MEANS, config.PIXEL_STDS, config.PIXEL_SCALE)
    # processed_ims.append(im_tensor)
    # im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
    # new_rec['im_info'] = np.array(im_info, dtype=np.float32)
    # processed_roidb.append(new_rec)





    # def get_roidb(self):
    #     # roidb = []
        
    #     for k, v in self.image_info.items():
    #         image_path = osp.join(self.data_path, k)
    #         image = cv2.imread(image_path) # (H, W, C)

    #         boxes = []
    #         landmarks = []
    #         gt_classes = []
    #         blur = []

    #         for line in v:
    #             values = [float(val) for val in line.split()]
    #             x1, y1, w, h = values[: 4]
                
    #             # filter by bounding box
    #             if x1<0 or y1<0 or w<0 or h<0:
    #                 continue
    #             if w<=cfg.MIN_FACE or h<=cfg.MIN_FACE:
    #                 continue

    #             box = [x1, y1, x1+w, y1+w]
    #             landmark = values[4: 19]
    #             blur_val = values[-1]
    #             gt_class = 1

    #             boxes.append(box)
    #             landmarks.append(landmark)
    #             blur.append(blur_val)
    #             gt_classes.append(gt_class)
    #         boxes = np.array(boxes, dtype=np.float32)
    #         landmarks = np.array(landmarks, dtype=np.float32).reshape(-1, 3)
    #         blur = np.array(blur, dtype=np.float32)
    #         gt_classes = np.array(gt_classes, dtype=np.int)

    #         roi = {
    #             'image_path': image_path, 
    #             'height': image.shape[0],
    #             'width': image.shape[1],
    #             'boxes': boxes,   # x1, y1, x2, y2
    #             'gt_classes': gt_classes,
    #             'blur': blur,
    #             'flipped': False
    #         }
    #         if len(boxes)>0:
    #             self.roidb.append(roi)
    #             print(roi)
    #         else:
    #             print(roi['image_path'])
    #     # print(roidb)
    #     print("roidb: ", len(self.roidb))
    #     # return roidb

def brightness_aug(src, x):
  alpha = 1.0 + random.uniform(-x, x)
  src *= alpha
  return src

def contrast_aug(src, x):
  alpha = 1.0 + random.uniform(-x, x)
  coef = np.array([[[0.299, 0.587, 0.114]]])
  gray = src * coef
  gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
  src *= alpha
  src += gray
  return src

def saturation_aug(src, x):
  alpha = 1.0 + random.uniform(-x, x)
  coef = np.array([[[0.299, 0.587, 0.114]]])
  gray = src * coef
  gray = np.sum(gray, axis=2, keepdims=True)
  gray *= (1.0 - alpha)
  src *= alpha
  src += gray
  return src

def color_aug(img, x):
  if cfg.COLOR_MODE>1:
    augs = [brightness_aug, contrast_aug, saturation_aug]
    random.shuffle(augs)
  else:
    augs = [brightness_aug]
  for aug in augs:
    #print(img.shape)
    img = aug(img, x)
    #print(img.shape)
  return img


if __name__ == "__main__":
    WiderFaceDetection(root_path="/Users/zhubin/Documents/work_git/retinaface_gt_v1.1/", 
                    data_path="/Users/zhubin/Documents/数据集/wider_face/WIDER_train/")




_ratio = (1.,)

RAC_SSH = {
    '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
}
