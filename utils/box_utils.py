import numpy as np
import torch
import time


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
    inter = torch.clamp((max_xy - min_xy), min=0)
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
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def box_overlaps(anchors, gt_boxes, mxnet_overlap=True):
    if mxnet_overlap:
        overlaps = box_overlaps_mxnet(anchors, gt_boxes)
    else:
        overlaps = jaccard(anchors, gt_boxes)

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







DEBUG = False
DEBUG = True

if DEBUG:
    anchors = np.array([[-248., -248.,  263.,  263.], 
                        [-216., -248.,  295.,  263.]])
    gt_boxes = np.array([183.12088, -47.304028, 478.5055, 306.68866]).reshape(1, 4)

    t1 = time.time()
    for i in range(10000):
        anchors = torch.Tensor(anchors)
        gt_boxes = torch.Tensor(gt_boxes)
        overlaps = box_overlaps(anchors, gt_boxes)
        # overlaps = box_overlaps_mxnet(anchors, gt_boxes)
    t2 = time.time()
    print(overlaps)
    print(t2 - t1)

    # array([0.07358106])
    # array([0.10577289])