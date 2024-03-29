import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url



# ResNet V1
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """  1x1 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out
        
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64)) * groups   
        self.conv1 = conv1x1(in_planes, width)   
        self.bn1 = norm_layer(width)   
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  
        self.bn2 = norm_layer(width) 
        self.conv3 = conv1x1(width, planes * self.expansion)  
        self.bn3 = norm_layer(planes * self.expansion) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, 
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=3, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )
        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, 
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def init_weights_torch_official(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        c3 = x
        x = self.layer3(x)
        c4 = x
        x = self.layer4(x)
        c5 = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # return [c2, c3, c4, c5]
        # print(c3.shape, c4.shape, c5.shape)
        return (c3, c4, c5)
        # return {"c3": c3, "c4": c4, "c5": c5}
        # return x

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
    # return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
    # return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained=False, progress=True)

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
    # return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained=False, progress=True)

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])
    # return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained=False, progress=True)

def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
    # return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained=False, progress=True)





# def _resnet(arch, block, layers, pretrained, progress, model_path=None):
#     model = ResNet(block, layers)
#     # print(model)
#     if model_path is not None:
#         state_dict = torch.load(model_path)
#     else:
#         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
#         # print(state_dict)
#     model.load_state_dict(state_dict)
#     return model

# def resnet18_old(pretrained=False, progress=True, model_path=None):
#     return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)

# def resnet34_old(pretrained=False, progress=True, model_path=None):
#     return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained=False, progress=True)

# def resnet50_old(pretrained=False, progress=True, model_path=None):
#     return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained=False, progress=True)

# def resnet101_old(pretrained=False, progress=True, model_path=None):
#     return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained=False, progress=True)

# def resnet152_old(pretrained=False, progress=True, model_path=None):
#     return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained=False, progress=True)

# def init_model(model, pretrained_model_path):
#     backbone_weights = torch.load(pretrained_model_path)
#     model.load_state_dict(backbone_weights)


    

# import cv2 
# import numpy as np
# from torchvision import models
# if __name__ == "__main__":

#     # img = cv2.imread("/home/dc2-user/zhubin/wider_face/train/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_893.jpg")
#     img = cv2.imread("/home/dc2-user/zhubin/wider_face/train/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_893.jpg")
#     img = cv2.resize(img, (640, 640)).astype(np.float32)
#     # img = img[:, :, (2, 1, 0)] # bgr 2 rgb
#     img = img.transpose(2, 0, 1) # (H,W,C) => (C,H,W)
#     img = torch.from_numpy(img).unsqueeze(0).cuda()

#     model = resnet50().cuda().train()
#     y = model(img)
#     print("out")
#     print(y[0].shape)
#     print(y[1].shape)
#     print(y[2].shape)

#     # print(max(y))

#     # model = models.resnet50(pretrained=True).eval().cuda() 
#     # y = model(img)
#     # print(torch.max(y))