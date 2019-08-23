import torch
import torch.nn.functional as F
import numpy as np

from easydict import EasyDict as edict
__C = edict()
cfg = __C

cfg.FACE_LANDMARK = True
cfg.SCALES = (640, 640)
cfg.CLIP = False
cfg.RPN_FEAT_STRIDE = [8, 16, 32]

cfg.RPN_ANCHOR = {
    '32': {'BASE_SIZE': 16, 'RATIOS': (1.0, ), 'SCALES': (32,16), 'FEAT_MAP_SIZE': [20, 20]},
    '16': {'BASE_SIZE': 16, 'RATIOS': (1.0, ), 'SCALES': (8,4), 'FEAT_MAP_SIZE': [40, 40]},
    '8':  {'BASE_SIZE': 16, 'RATIOS': (1.0, ), 'SCALES': (2,1), 'FEAT_MAP_SIZE': [80, 80]},
}



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



def bbox_pred(boxes_anchor, boxes_deltas):

    widths = boxes_anchor[:, 2] - boxes_anchor[:, 0] + 1.0
    heights = boxes_anchor[:, 3] - boxes_anchor[:, 1] + 1.0
    a_cx = boxes_anchor[:, 0] + 0.5 * (widths - 1.0)
    a_cy = boxes_anchor[:, 1] + 0.5 * (heights - 1.0)

    dx = boxes_deltas[:, 0]
    dy = boxes_deltas[:, 1]
    dw = boxes_deltas[:, 2]
    dh = boxes_deltas[:, 3]

    pred_cx = (dx * widths) + a_cx
    pred_cy = (dy * heights) + a_cy
    pred_dw = torch.exp(dw) * widths
    pred_dh = torch.exp(dh) * heights

    pred_xmin = pred_cx - 0.5 * (pred_dw - 1.0)
    pred_ymin = pred_cy - 0.5 * (pred_dh - 1.0)
    pred_xmax = pred_cx + 0.5 * (pred_dw - 1.0)
    pred_ymax = pred_cy + 0.5 * (pred_dh - 1.0)

    boxes_pred = torch.cat((pred_xmin, pred_ymin, pred_xmax, pred_ymax), dim=-1)
    return boxes_pred

def landmark_pred(boxes_anchor, landmark_deltas):
    widths = boxes_anchor[:, 2] - boxes_anchor[:, 0] + 1.0
    heights = boxes_anchor[:, 3] - boxes_anchor[:, 1] + 1.0
    a_cx = boxes_anchor[:, 0] + 0.5 * (widths - 1.0)
    a_cy = boxes_anchor[:, 1] + 0.5 * (heights - 1.0)

    l_pred_x1 = landmark_deltas[:, 0] * widths + a_cx
    l_pred_y1 = landmark_deltas[:, 1] * heights + a_cy
    l_pred_x2 = landmark_deltas[:, 2] * widths + a_cx
    l_pred_y2 = landmark_deltas[:, 3] * heights + a_cy
    l_pred_x3 = landmark_deltas[:, 4] * widths + a_cx
    l_pred_y3 = landmark_deltas[:, 5] * heights + a_cy
    l_pred_x4 = landmark_deltas[:, 6] * widths + a_cx
    l_pred_y4 = landmark_deltas[:, 7] * heights + a_cy
    l_pred_x5 = landmark_deltas[:, 8] * widths + a_cx
    l_pred_y5 = landmark_deltas[:, 9] * heights + a_cy
    
    landmark_pred = torch.stack((l_pred_x1, l_pred_y1,
                                 l_pred_x2, l_pred_y2,
                                 l_pred_x3, l_pred_y3,
                                 l_pred_x4, l_pred_y4,
                                 l_pred_x5, l_pred_y5), 1)
    return landmark_pred



class Detect(Function):

    def __init__(self, ):
        self.num_classes = 2
        # self.base_anchors_num = []
        # for k, v in enumerate(cfg.RPN_FEAT_STRIDE):
        #     s_stride = str(v)
        #     base_anchors_num_feat = pow(2, len(cfg.RPN_ANCHOR[s_stride]['RATIOS'])-1) * \
        #                             pow(2, len(cfg.RPN_ANCHOR[s_stride]['SCALES'])-1)
        #     self.base_anchors_num.appennd(base_anchors_num_feat)
        self.feat_strides = cfg.RPN_FEAT_STRIDE
        self.rpn_anchor = cfg.RPN_ANCHOR 
        

# base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, np.float32), stride=feat_stride)
# all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
# anchors_plane(feat_height, feat_width, feat_stride, base_anchors):

    def forward(self, predictions):
        if cfg.FACE_LANDMARK:
            conf_pred_batch, loc_pred_batch, landmark_pred_batch = predictions
        else:
            conf_pred_batch, loc_pred_batch = predictions
        # conf_pred_batch = torch.cat((conf_pred_batch[0], conf_pred_batch[1], conf_pred_batch[2]), dim=1)
        # loc_pred_batch = torch.cat((loc_pred_batch[0], loc_pred_batch[1], loc_pred_batch[2]), dim=1)
        # if cfg.FACE_LANDMARK:
        #     landmark_pred_batch = torch.cat((landmark_pred_batch[0], landmark_pred_batch[1], landmark_pred_batch[2]), dim=1)

        # 对于不同大小的输入，batch只能为1
        batch = conf_pred_batch[0].size(0)
        assert batch == 1  # TODO

        anchors_list = []
        anchors_total_num = 0

        conf_list = list()
        loc_list = list()
        if cfg.FACE_LANDMARK:
            landmark_list = list()

        for i in range(len(self.feat_strides)):
            # TODO  need to change retina_face changed multitask need to change
            
            feat_stride = self.feat_strides[i]
            stride_str = str(feat_stride)
            base_size = self.rpn_anchor[stride_str]['BASE_SIZE']
            ratios = self.rpn_anchor[stride_str]['RATIOS']
            scales = self.rpn_anchor[stride_str]['SCALES']
            # feat_height, feat_width = self.rpn_anchor[stride_str]['FEAT_MAP_SIZE']
            feat_height, feat_width = conf_pred_batch[i].size(1), conf_pred_batch[i].size(2)

            base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, np.float32), stride=feat_stride)
            # print(base_anchors)
            feat_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)

            # Decode
            scores = F.softmax(conf_pred_batch[i].view(-1, self.num_classes))
            decoded_boxes = bbox_pred(feat_anchors.view(-1, 4), loc_pred_batch[i].view(-1, 4))
            if cfg.FACE_LANDMARK:
                decoded_landmarks = landmark_pred(feat_anchors.view(-1, 4), landmark_pred_batch[i].view(-1, 10))  # TODO

            conf_list.append(scores)
            loc_list.append(decoded_boxes)
            if cfg.FACE_LANDMARK:
                landmark_list.append(decoded_landmarks)
        conf_pred = torch.cat(conf_list, dim=0)
        loc_pred = torch.cat(loc_list, dim=0)
        if cfg.FACE_LANDMARK:
            landmark_pred = torch.cat(landmark_list, dim=0)
            return conf_pred, loc_pred, landmark_pred
        return conf_pred, loc_pred






        # A = base_anchors.shape[0]
        # anchors_num = A * feat_height * feat_width # TODO
        # all_anchors = all_anchors.reshape(anchors_nums, 4)  # TODO
        # anchors_list.append(all_anchors)

        # stride = cfg.RPN_FEAT_STRIDE[i]
        # s_stride = str(stride)
        # base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, np.float32), stride=feat_stride)
        # # all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)

        # base_anchors_num = self.base_anchors_num[s_stride]

        # anchors_num_feat = conf_pred_batch[i].size(1)  # // base_anchors_num
        # anchors_total_num += anchors_num_feat
        # ###

        # base_anchor_nums = cfg.RPN_ANCHOR[]
        
        # conf_pred_batch[i].size()
        # self.scores = conf_pred_batch[i]

        # anchors_num = anchors.size(0)

        # self.scores = torch.zeros(batch, anchors_num, self.num_classes)
        # self.boxes = torch.zeros(batch, anchors_num, 4)
        # if cfg.FACE_LANDMARK:
        #     self.landmarks = torch.zeros(1, anchors_num, 10)
        # if conf_pred_batch[0].is_cuda:   # is merge  TODO
        #     self.scores = self.scores.cuda()
        #     self.boxes = self.boxes.cuda()
        #     if cfg.FACE_LANDMARK:
        #         self.landmarks = self.landmarks.cuda()
        # ###         

        # # out = torch.cat([x1, x2, x3], dim=1)   ???????
        # conf_pred = torch.stack(conf_pred_batch)
        # conf = F.softmax(conf_pred.view(-1, self.num_classes), 1)

