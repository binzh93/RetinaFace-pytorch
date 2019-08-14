#-*- coding: utf-8 -*-
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import cv2
import torch
from .data_augment import *
from .roidb import get_roidb

# landmark  -1: no landmark
# landmark  1.0: occlusion 
# landmark 0.0: normal
from easydict import EasyDict as edict
__C = edict()
cfg = __C

# cfg.Landmark = True
cfg.FACE_LANDMARK = True
cfg.MIN_FACE = 0
cfg.USE_FLIPED = True


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
        # txt_file = osp.join(self.root_path, phase, "label_less.txt")
        txt_file = osp.join(self.root_path, phase, "label.txt")

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
        return image, boxes, landmarks
        
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
        roi = crop_img(roi)
#         print("LLLLLLL: ", type(roi['landmarks']))

        # if self.transform is not None:
        #     roi = self.transform(roi)
        if 'image_data' in roi:
            image =  roi['image_data']
        else:
            image = cv2.imread(roi['image_path'])
#         if roi['flipped']:
#             image = image[:, ::-1]
        isColor_JITTERING = False
        if isColor_JITTERING:
            pass
        image = image.astype(np.float32)
        PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
        PIXEL_STDS = np.array([1.0, 1.0, 1.0])
        PIXEL_SCALE = 1.0
        image = transform_ms(image, PIXEL_MEANS, PIXEL_STDS, PIXEL_SCALE) # already to NCHW
        image = image.astype(np.float32) # TODO if must ????
        # image = roi['image_data']
        # image = cv2.imread(roi['image_path'])
        # print("image_data is in roi: ", roi['image_data'].shape)
        # image = roi['image_data']

        # image = image[:, :, (2, 1, 0)]  # to rgb
        im_tensor = torch.from_numpy(image)#.permute(2, 0, 1)
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




if __name__ == "__main__":
    WiderFaceDetection(root_path="/Users/zhubin/Documents/work_git/retinaface_gt_v1.1/", 
                    data_path="/Users/zhubin/Documents/数据集/wider_face/WIDER_train/")




_ratio = (1.,)

RAC_SSH = {
    '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
}
