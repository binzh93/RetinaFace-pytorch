#-*- coding: utf-8 -*-
import numpy as np
import torch
import time
from torch.autograd import Variable

from easydict import EasyDict as edict
__C = edict()
cfg = __C
# cfg.RPN_NEGATIVE_OVERLAP = 0.3
# cfg.RPN_POSITIVE_OVERLAP = 0.5

# cfg.RPN_ENABLE_OHEM = 1
# cfg.FACE_LANDMARK = True

# cfg.USE_BLUR = False
cfg.USE_OCCLUSION = False

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0) + 1

    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]+1) *
              (box_a[:, 3]-box_a[:, 1]+1)).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]+1) *
              (box_b[:, 3]-box_b[:, 1]+1)).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def box_overlaps(gt_boxes, anchors, mxnet_overlap=False):
    if mxnet_overlap:
        overlaps = box_overlaps_mxnet(anchors, gt_boxes)
    else:
        overlaps = jaccard(gt_boxes, anchors)

    return overlaps

def box_overlaps_mxnet(anchors, gt_boxes):
    '''
    About Ten times faster than torch process method like jaccard
    '''
    N = anchors.shape[0]
    K = gt_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = (
            (gt_boxes[k, 2] - gt_boxes[k, 0] + 1) *
            (gt_boxes[k, 3] - gt_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(anchors[n, 2], gt_boxes[k, 2]) -
                max(anchors[n, 0], gt_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(anchors[n, 3], gt_boxes[k, 3]) -
                    max(anchors[n, 1], gt_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (anchors[n, 2] - anchors[n, 0] + 1) *
                        (anchors[n, 3] - anchors[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bbox_transform2(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.size(0) == gt_rois.size(0), 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)
    # print(targets_dx.shape)
    # print(targets_dy.shape)
    # print(targets_dw.shape)
    # print(targets_dh.shape)

    if gt_rois.shape[1]<=4:
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), -1)
        return targets
    else:
        raise NotImplementedError
    

def nonlinear_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    if gt_rois.shape[1]<=4:
      targets = np.vstack(
          (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
      return targets
    else:
      targets = [targets_dx, targets_dy, targets_dw, targets_dh]
      #if config.USE_BLUR:
      #  for i in range(4, gt_rois.shape[1]):
      #    t = gt_rois[:,i]
      #    targets.append(t)
      targets = np.vstack(targets).transpose()
      return targets

def landmark_transform2(ex_rois, gt_rois):
    assert ex_rois.size(0) == gt_rois.size(0), 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    if not cfg.USE_OCCLUSION:
        if len(gt_rois.size()) == 2:
            landmark_len = 10
        else:
            landmark_len = 15
        gt_rois = gt_rois.view(-1, landmark_len)

        landmark_dx1 = (gt_rois[:, 0] - ex_ctr_x) / (ex_widths + 1e-14)
        landmark_dy1 = (gt_rois[:, 1] - ex_ctr_y) / (ex_heights + 1e-14)
        landmark_dx2 = (gt_rois[:, 2] - ex_ctr_x) / (ex_widths + 1e-14)
        landmark_dy2 = (gt_rois[:, 3] - ex_ctr_y) / (ex_heights + 1e-14)
        landmark_dx3 = (gt_rois[:, 4] - ex_ctr_x) / (ex_widths + 1e-14)
        landmark_dy3 = (gt_rois[:, 5] - ex_ctr_y) / (ex_heights + 1e-14)
        landmark_dx4 = (gt_rois[:, 6] - ex_ctr_x) / (ex_widths + 1e-14)
        landmark_dy4 = (gt_rois[:, 7] - ex_ctr_y) / (ex_heights + 1e-14)
        landmark_dx5 = (gt_rois[:, 8] - ex_ctr_x) / (ex_widths + 1e-14)
        landmark_dy5 = (gt_rois[:, 9] - ex_ctr_y) / (ex_heights + 1e-14)

    
    elif (gt_rois.size() == 3) and cfg.USE_OCCLUSION:
        pass # TODO

    targets_landmark = torch.stack((landmark_dx1, landmark_dy1, 
                                    landmark_dx2, landmark_dy2,
                                    landmark_dx3, landmark_dy3,
                                    landmark_dx4, landmark_dy4,
                                    landmark_dx5, landmark_dy5), 1)  # (anchors_num, 10)
    # print("landmark: ", targets_landmark.shape)
    return targets_landmark



def landmark_transform(ex_rois, gt_rois):

    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    
    targets = []
    for i in range(gt_rois.shape[1]):
      for j in range(gt_rois.shape[2]):
        #if not config.USE_OCCLUSION and j==2:
        #  continue
        if j==2:
          continue
        if j==0: #w
          target = (gt_rois[:,i,j] - ex_ctr_x) / (ex_widths + 1e-14)
        elif j==1: #h
          target = (gt_rois[:,i,j] - ex_ctr_y) / (ex_heights + 1e-14)
        else: #visibile
        #   target = gt_rois[:,i,j]
          raise "error"
        targets.append(target)


    targets = np.vstack(targets).transpose()
    return targets

bbox_transform = nonlinear_transform



if False:
    gt_boxes = np.array([[183.12088, -47.304028, 478.5055, 306.68866],
                         [183.12088, -47.304028, 478.5055, 306.68866]
                         ]).reshape(-1, 4)
    gt = np.array([gt_boxes.copy() for i in range(5)]).reshape(-1, 4)
    # print(gt_boxes)
    # print(g_t)
    gt_2 = gt.copy()

    gt_list1 = []
    t1 = time.time()
    for i in range(10000):
        for _, sample in enumerate(gt):
            gt_list1.append(torch.FloatTensor(sample))
    t2 = time.time()
    print(t2 - t1)

    gt_list2 = []
    t1 = time.time()
    for i in range(10000):
        for _, sample in enumerate(gt_2):
            gt_list2.append(torch.from_numpy(sample).float())
    t2 = time.time()
    print(t2 - t1)




#         targets.append(torch.FloatTensor(sample[1]))
# #         landmarks.append(torch.FloatTensor(sample[2]))
#         boxes.append(torch.FloatTensor(sample[1]))
#         landmarks.append(torch.FloatTensor(sample[2]))



DEBUG = False
# DEBUG = True

if DEBUG:
    anchors = np.array([[-248., -248.,  263.,  263.], 
                        [-216., -248.,  295.,  263.]])
    gt_boxes = np.array([183.12088, -47.304028, 478.5055, 306.68866]).reshape(1, 4)
    
    anchors = np.array([anchors for i in range(8400)], dtype=np.float32).reshape(-1, 4)
    gt_boxes = np.array([gt_boxes for i in range(1)], dtype=np.float32).reshape(-1, 4)

    a_t = np.array([anchors for i in range(32)]).reshape(32, -1 , 4)
    b_t = np.array([gt_boxes for i in range(32)]).reshape(32, -1, 4)

    # a_t = np.array([anchors for i in range(10)]).reshape(10, -1 , 4)
    # b_t = np.array([gt_boxes for i in range(10)]).reshape(10, -1, 4)
    anchors = a_t
    gt_boxes = b_t
    
    batch_size = a_t.shape[0]  # bs = 32

    print("copy from mxnet")
    print(anchors.shape)
    print(gt_boxes.shape)
    t1 = time.time()
    for i in range(1):
#         anchors = torch.Tensor(anchors)
#         gt_boxes = torch.Tensor(gt_boxes)
#         overlaps = box_overlaps(anchors, gt_boxes)
        for j in range(batch_size):
            overlaps = box_overlaps_mxnet(anchors[j], gt_boxes[j])
        print(overlaps.shape)
    print("overlap:", overlaps.shape)
    t2 = time.time()
    print(overlaps)
    print(t2 - t1)
    
    
    print("SSD method")
    t_total = 0.0
    
    
#     aa = [Variable(torch.Tensor(anchors.copy()), requires_grad=False).cuda() for i in range(1000)]
#     gg = [Variable(torch.Tensor(gt_boxes.copy()), requires_grad=False).cuda() for i in range(1000)]
    t222 = time.time()
    # aa = Variable(torch.Tensor(anchors.copy()), requires_grad=False).cuda()
    aa = torch.Tensor(anchors.copy()).cuda()
    t33 = time.time()
    gg = torch.Tensor(gt_boxes.copy()).cuda()
#     gg = Variable(torch.Tensor(gt_boxes.copy()), requires_grad=False).cuda()

    print("convert tensor and cuda1: ", t33 - t222)
    print("convert tensor and cuda2: ", time.time() - t33)

    t3 = time.time()
    for i in range(1):
#         anchors_ = torch.Tensor(anchors)
#         gt_boxes_ = torch.Tensor(gt_boxes)
#         anchors_ = Variable(anchors_, requires_grad=False).cuda()
#         gt_boxes_ = Variable(gt_boxes_, requires_grad=False).cuda()
#         t1 = time.time()
        for j in range(batch_size):
            overlaps = box_overlaps(aa[j], gg[j])
        # overlaps = box_overlaps_mxnet(anchors, gt_boxes)
        t2 = time.time()
#         t_total += (t2-t1)
        print(overlaps.shape)
#     t2 = time.time()
    t4 =time.time()
    print(overlaps)
#     print(t_total)
    print(t4-t3)

    # array([0.07358106])
    # array([0.10577289])