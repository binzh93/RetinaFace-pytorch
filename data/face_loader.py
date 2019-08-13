import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import cv2

from roidb import get_roidb

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


class FaceTransform(object):
    def __init__(self):
        pass

    def __call__(self, target, landmark, width, height):
        boxes = []
        landmarks = []
        labels = 1
        return 



class WiderFaceDetection(data.Dataset):

    def __init__(self, root_path, data_path, phase="train", dataset_name="WiderFace", transform=None):
        self.name = dataset_name
        self.root_path = root_path
        self.target_transform = FaceTransform()

        self.data_path = osp.join(data_path, "images")
        txt_file = osp.join(self.root_path, phase, "label.txt")

        self.image_info = {}
        self.load_info(txt_file)

        self.roidb = get_roidb(self.image_info, self.data_path)
        print("WiderFaceDetection:", len(self.roidb))
        # self.get_roidb()
        
    
    def __getitem__(self, index):
        pull_item(index)
        



        # if cfg.FACE_LANDMARK:
        #     image, boxes, landmark = self.pull_item(index)
        # else:
        #     NotImplementedError  # TODO
        # return image, boxes, landmark
        
    
    def __len__(self):
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

        if self.transform is not None:
            roi = self.transform(roi)
        return roi['']



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
