import torch.nn as nn
import numpy as np
import cv2
from models.model_helper import FPN, ContextModule
from models.retina_face import *
# from model_helper import FPN, ContextModule
# from retina_face import *

from easydict import EasyDict as edict
__C = edict()
cfg = __C

cfg.USE_BLUR = False
cfg.FACE_LANDMARK = True
cfg.USE_OCCLUSION = False




class RetinaFace(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super(RetinaFace, self).__init__()
        # self.res = 
        self.num_classes = num_classes
        self.features = backbone
        self.fpn = FPN(channel=[512, 1024, 2048])

        self.context_module1 = ContextModule(in_channels=256)
        self.context_module2 = ContextModule(in_channels=256)
        self.context_module3 = ContextModule(in_channels=256)

        num_anchors = 2 # TODO
        self.bbox_pred_len = 4 # TODO
        self.landmark_pred_len = 10 # TODO
        if cfg.USE_BLUR:
            self.bbox_pred_len = 5
        if cfg.USE_OCCLUSION:
            self.landmark_pred_len = 15

        self.rpn_cls_score = nn.Conv2d(in_channels=256, out_channels=num_anchors*self.num_classes, kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(in_channels=256, out_channels=num_anchors*self.bbox_pred_len, kernel_size=1)
        if cfg.FACE_LANDMARK:
            self.rpn_landmark_pred = nn.Conv2d(in_channels=256, out_channels=num_anchors*self.landmark_pred_len, kernel_size=1)
        
        # self.conv_cls = nn.Sequential()
        # self.conv_loc = nn.Sequential()
        # if cfg.FACE_LANDMARK:
        #     self.conv_landmark = nn.Conv2d(in_channels=, )
    
    def forward(self, x):
        c3, c4, c5 = self.features(x)
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

        for fea in fea_fpn:
            rpn_cls_score = self.rpn_cls_score(fea).permute(0, 2, 3, 1).contiguous()
            rpn_bbox_pred = self.rpn_bbox_pred(fea).permute(0, 2, 3, 1).contiguous()
            conf_pred.append(rpn_cls_score.view(rpn_cls_score.size(0), -1, self.num_classes))
            loc_pred.append(rpn_bbox_pred.view(rpn_bbox_pred.size(0), -1, self.bbox_pred_len))
            if cfg.FACE_LANDMARK:
                rpn_landmark_pred = self.rpn_landmark_pred(fea).permute(0, 2, 3, 1).contiguous()
                landmarks_pred.append(rpn_landmark_pred.view(rpn_landmark_pred.size(0), -1, self.landmark_pred_len))
        out = (conf_pred, loc_pred, landmarks_pred) 
        return out
       

        # rpn_cls_score_list = []
        # rpn_bbox_pred_list = []
        # if cfg.Landmark:
        #     rpn_landmark_pred = [] 
        



        # fea_fpn = [m1, m2, m3]
        # loc_pre = []
        # for i in range(3):
        #     rpn_cls_score = 
        #     loc_pre.append()

        #     for (x,l,c) in zip(sources, self.loc, self.conf):
        #         loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        #         conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        # return {8: m1, 16: m2, 32: m3}
        # out = (
            
        # )


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

    