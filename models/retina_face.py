import torch
import torch.nn as nn
import numpy as np
import cv2
from models.model_helper import FPN, ContextModule, init_all_layers, initialize_layer
# from models.retina_face import *
# from model_helper import FPN, ContextModule
# from retina_face import *


from easydict import EasyDict as edict
__C = edict()
cfg = __C

cfg.USE_BLUR = False
cfg.FACE_LANDMARK = True
cfg.USE_OCCLUSION = False


# import torch.nn.init as init

# def weights_init(m):
#     for key in m.state_dict():
#         if key.split('.')[-1] == 'weight':
#             if 'conv' in key:
#                 init.kaiming_normal(m.state_dict()[key], mode='fan_out')
#             if 'bn' in key:
#                 m.state_dict()[key][...] = 1
#         elif key.split('.')[-1] == 'bias':
#             m.state_dict()[key][...] = 0


class RetinaFace(nn.Module):
    def __init__(self, backbone, num_classes=2, pretrained_model_path=None):
        super(RetinaFace, self).__init__()
        # self.res = 
        self.num_classes = num_classes
        self.backbone = backbone
        self.fpn = FPN(channel=[512, 1024, 2048])
        self.pretrained_model_path = pretrained_model_path

        self.context_module1 = ContextModule(in_channels=256)
        self.context_module2 = ContextModule(in_channels=256)
        self.context_module3 = ContextModule(in_channels=256)

        # num_anchors = 2 # TODO
        self.base_anchors_num = 2 # TODO
        self.bbox_pred_len = 4 # TODO
        self.landmark_pred_len = 10 # TODO
        if cfg.USE_BLUR:
            self.bbox_pred_len = 5
        if cfg.USE_OCCLUSION:
            self.landmark_pred_len = 15

        self.rpn_cls_score = nn.Conv2d(in_channels=256, out_channels=self.base_anchors_num*self.num_classes, kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(in_channels=256, out_channels=self.base_anchors_num*self.bbox_pred_len, kernel_size=1)
        if cfg.FACE_LANDMARK:
            self.rpn_landmark_pred = nn.Conv2d(in_channels=256, out_channels=self.base_anchors_num*self.landmark_pred_len, kernel_size=1)
        
        self._init_modules_()

    def _init_modules_(self):
        self.rpn_cls_score.apply(initialize_layer)
        self.rpn_bbox_pred.apply(initialize_layer)
        if cfg.FACE_LANDMARK:
            self.rpn_landmark_pred.apply(initialize_layer)
        if self.pretrained_model_path:
            print("load pretrained model...")
            backbone_weights = torch.load(self.pretrained_model_path)
            self.backbone.load_state_dict(backbone_weights)
    
    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        # print(c3.shape)
        # print(c4.shape)
        # print(c5.shape)
        p3, p4, p5 = self.fpn([c3, c4, c5])
        # print(p3.shape)
        # print(p4.shape)
        # print(p5.shape)
        m1 = self.context_module1(p3)
        m2 = self.context_module1(p4)
        m3 = self.context_module1(p5)
        # print(m1.shape)
        # print(m2.shape)
        # print(m3.shape)

        conf_pred = list()
        loc_pred = list()
        if cfg.FACE_LANDMARK:
            landmarks_pred = list()
        fea_fpn = [m1, m2, m3]

        # TODO Raw method 
#         for fea in fea_fpn:
#             rpn_cls_score = self.rpn_cls_score(fea).permute(0, 2, 3, 1).contiguous()
#             conf_pred.append(rpn_cls_score.view(rpn_cls_score.size(0), -1, self.num_classes))

#             rpn_bbox_pred = self.rpn_bbox_pred(fea).permute(0, 2, 3, 1).contiguous()
#             loc_pred.append(rpn_bbox_pred.view(rpn_bbox_pred.size(0), -1, self.bbox_pred_len))
#             if cfg.FACE_LANDMARK:
#                 rpn_landmark_pred = self.rpn_landmark_pred(fea).permute(0, 2, 3, 1).contiguous()
#                 landmarks_pred.append(rpn_landmark_pred.view(rpn_landmark_pred.size(0), -1, self.landmark_pred_len))
        for fea in fea_fpn:
            rpn_cls_score = self.rpn_cls_score(fea).permute(0, 2, 3, 1).contiguous()
            conf_pred.append(rpn_cls_score)

            rpn_bbox_pred = self.rpn_bbox_pred(fea).permute(0, 2, 3, 1).contiguous()
            loc_pred.append(rpn_bbox_pred)
            if cfg.FACE_LANDMARK:
                rpn_landmark_pred = self.rpn_landmark_pred(fea).permute(0, 2, 3, 1).contiguous()
                landmarks_pred.append(rpn_landmark_pred)
        
        # return conf_pred
        if cfg.FACE_LANDMARK:
            out = (conf_pred, loc_pred, landmarks_pred) 
            return out
        else:
            out = (conf_pred, loc_pred) 
            return out

       
        


if __name__ == "__main__":
    img = cv2.imread("/Users/zhubin/Documents/work_git/RetinaFace-pytorch/images/tensorrt_python_support.png")
    img = cv2.resize(img, (640, 640)).astype(np.float32)
    img = img[:, :, (2, 1, 0)] # bgr 2 rgb
    img = img.transpose(2, 0, 1) # (H,W,C) => (C,H,W)
    img = torch.from_numpy(img).unsqueeze(0)#.cuda()

    backbone = resnet50()
    model = RetinaFace(backbone)

    # fea = resnet50()#.eval()#.cuda()
    # model = FPN(fea)
    # print(model)
    # fea_dict = model(img)
    conf_pred, loc_pred, landmarks_pred = model(img)
    print(conf_pred[0].shape)
    print(conf_pred[1].shape)
    print(conf_pred[2].shape)
    print("==========")
    print(loc_pred[0].shape)
    print(loc_pred[1].shape)
    print(loc_pred[2].shape)
    print("==========")
    print(landmarks_pred[0].shape)
    print(landmarks_pred[1].shape)
    print(landmarks_pred[2].shape)

    