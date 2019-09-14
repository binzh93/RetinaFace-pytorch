
from easydict import EasyDict as edict
__C = edict()
cfg = __C


cfg.USE_BLUR = False
cfg.FACE_LANDMARK = False
cfg.USE_OCCLUSION = False
# cfg.PRETRAIN_WEIGHTS = "weights/pretrained/resnet18_official_pretrain.pth"
cfg.PRETRAIN_WEIGHTS = "weights/pretrained/resnet50_official_pretrain.pth"
cfg.ARCH = "resnet50"


