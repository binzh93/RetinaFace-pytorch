import torch 
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
# from torchvision import models

from easydict import EasyDict as edict
__C = edict()
cfg = __C

cfg.USE_CROP = True
# cfg.FACE_LANDMARK = True
# cfg.USE_OCCLUSION = False



def conv_act_layer(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act_type=None):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    layers.append(nn.BatchNorm2d(out_channels))
    if act_type == "relu":
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextModule, self).__init__()
        self.in_channels = in_channels
        self.conv3x3 = conv_act_layer(self.in_channels, self.in_channels//2, kernel_size=3, padding=1, act_type='relu')

        self.conv_p = conv_act_layer(self.in_channels, self.in_channels//4, kernel_size=3, padding=1, act_type='relu')
        self.conv5x5 = conv_act_layer(self.in_channels//4, self.in_channels//4, kernel_size=3, padding=1)

        self.conv7x7_1 = conv_act_layer(self.in_channels//4, self.in_channels//4, kernel_size=3, padding=1, act_type='relu')
        self.conv7x7 = conv_act_layer(self.in_channels//4, self.in_channels//4, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

        init_all_layers(self.modules())

    def forward(self, x):
        x1 = self.conv3x3(x)

        x = self.conv_p(x)
        x2 = self.conv5x5(x)

        x3 = self.conv7x7_1(x)
        x3 = self.conv7x7(x3)

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, channel=[512, 1024, 2048]): # channel=[256, 512, 1024, 2048]
        super(FPN, self).__init__()
        # self.features = features

        self.channel = channel
        features_map = [80, 40, 20]
        self.lateral3 = conv_act_layer(self.channel[0], 256, kernel_size=1, act_type='relu')
        self.lateral4 = conv_act_layer(self.channel[1], 256, kernel_size=1, act_type='relu')
        self.lateral5 = conv_act_layer(self.channel[2], 256, kernel_size=1, act_type='relu')

        self.p3 = conv_act_layer(256, 256, kernel_size=3, padding=1, act_type='relu')
        self.p4 = conv_act_layer(256, 256, kernel_size=3, padding=1, act_type='relu')

        init_all_layers(self.modules())

    def forward(self, x):
        if len(x) == 3:
            c3, c4, c5 = x
        else:
            raise NotImplementedError
     
        c3_lateral = self.lateral3(c3)
        c4_lateral = self.lateral4(c4)
        c5_lateral = self.lateral5(c5)

        c5_up = F.interpolate(c5_lateral, scale_factor=2, mode='nearest')
        if cfg.USE_CROP:
            c4 = c4_lateral + c5_up[:, :, 0: c4_lateral.size(2), 0:c4_lateral.size(3)]
        else:
            raise NotImplementedError
        c4 = self.p4(c4)

        c4_up = F.interpolate(c4, scale_factor=2, mode='nearest')
        if cfg.USE_CROP:
            c3 = c3_lateral + c4_up[:, :, 0: c3_lateral.size(2), 0: c3_lateral.size(3)]
        else:
            raise NotImplementedError
        c3 = self.p3(c3)

        p3 = c3
        p4 = c4
        p5 = c5_lateral
        # print(p3.shape)
        # print(p4.shape)
        # print(p5.shape)
        return (p3, p4, p5)


def init_all_layers(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def initialize_layer(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    if isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1.0)
        nn.init.constant_(layer.bias, 0.0)
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.constant_(layer.bias, 0)