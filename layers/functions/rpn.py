from itertools import product
import numpy as np
import torch
# from itertools import product as product

from easydict import EasyDict as edict
__C = edict()
cfg = __C
cfg.SCALES = (640, 640)
cfg.CLIP = False
# cfg.SCALES = [(640, 640)]
# cfg.FACE_LANDMARK = True
# cfg.MIN_FACE = 0
# cfg.COLOR_JITTERING = 0.125
# cfg.USE_FLIPED = True

# cfg.FEATURE_MAPS = [[80, 80], [40, 40], [20, 20]]
# cfg.RPN_FEAT_STRIDE = [32, 16, 8]
cfg.RPN_FEAT_STRIDE = [8, 16, 32]
cfg.RPN_ANCHOR = {
    '32': {'BASE_SIZE': 16, 'RATIOS': (1.0, ), 'SCALES': (32,16), 'FEAT_MAP_SIZE': [20, 20]},
    '16': {'BASE_SIZE': 16, 'RATIOS': (1.0, ), 'SCALES': (8,4), 'FEAT_MAP_SIZE': [40, 40]},
    '8':  {'BASE_SIZE': 16, 'RATIOS': (1.0, ), 'SCALES': (2,1), 'FEAT_MAP_SIZE': [80, 80]},
}
# base_size = cfg.RPN_ANCHOR[stride_str]['BASE_SIZE']
# ratios = cfg.RPN_ANCHOR[stride_str]['RATIOS']
# scales = cfg.RPN_ANCHOR[stride_str]['SCALES']


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=np.array([32.0, 16.0]), stride=32):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
    return anchors

# a = generate_anchors(base_size=16, ratios=[1.0], scales=np.array([32.0, 16.0]), stride=32)
# a = generate_anchors(base_size=16, ratios=[1.0], scales=np.array([8.0, 4.0]), stride=16)
# a = generate_anchors(base_size=16, ratios=[1.0], scales=np.array([2.0, 1.0]), stride=8)
# a = generate_anchors(base_size=16, ratios=[1.0], scales=[32.0, 16.0], stride=32)

# print(a)

def anchors_plane(feat_height, feat_width, feat_stride, base_anchors):
    A = base_anchors.shape[0]
    all_anchors = np.zeros((feat_height, feat_width, A, 4))
    for ih in range(feat_height):
        h_off = ih * feat_stride
        for iw in range(feat_width):
            w_off = iw * feat_stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + w_off
                all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + h_off
                all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + w_off
                all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + h_off
    return all_anchors


class Anchor_Box(object):
    def __init__(self, ):
        super(Anchor_Box, self).__init__()
        # self.feature_maps = cfg.FEATURE_MAPS
        pass

    def forward(self):
        
        feat_strides = cfg.RPN_FEAT_STRIDE
        # for k, f in enumerate(self.feature_maps):
        #     for i, j in product(range(f), repeat=2):
        #         base_anchors = generate_anchors(base_size=16, ratios=[1.0], scales=np.array([32.0, 16.0]), stride=32)
        anchors_list = []
        for i in range(len(feat_strides)):
            feat_stride = feat_strides[i]
            stride_str = str(feat_stride)
            base_size = cfg.RPN_ANCHOR[stride_str]['BASE_SIZE']
            ratios = cfg.RPN_ANCHOR[stride_str]['RATIOS']
            scales = cfg.RPN_ANCHOR[stride_str]['SCALES']
            feat_height, feat_width = cfg.RPN_ANCHOR[stride_str]['FEAT_MAP_SIZE']

            base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, np.float32), stride=feat_stride)
            all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
            A = base_anchors.shape[0]
            anchors_nums = A * feat_height * feat_width
            all_anchors = all_anchors.reshape(anchors_nums, 4)
            # allowed_border = 9999
            # inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) & 
            #                        (all_anchors[:, 1] >= -allowed_border) &
            #                        (all_anchors[:, 2] < cfg.SCALES[0] + allowed_border) & 
            #                        (all_anchors[:, 3] < cfg.SCALES[0] + allowed_border))[0]
            # anchors = all_anchors[inds_inside, :]
            anchors_list.append(all_anchors)
        anchors = np.concatenate(anchors_list)
        # print(anchors.shape)

        # TODO
        anchors_tensor = torch.Tensor(anchors).view(-1, 4)
        # if cfg.CLIP:
        #     anchors_t.clamp_(max=cfg.SCALES[0], min=0)
        # return anchors_t
        # print("rpn file: ", anchors_tensor.shape)
        return anchors_tensor


           

# a = Prior_Box()
# a.forward()




