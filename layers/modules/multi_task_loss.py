import torch
import torch.nn as nn
from utils.box_utils import box_overlaps, bbox_transform, landmark_transform
import numpy as np
import numpy.random as npr
import copy
import torch.nn.functional as F


from easydict import EasyDict as edict
__C = edict()
cfg = __C
cfg.RPN_NEGATIVE_OVERLAP = 0.3
config.RPN_POSITIVE_OVERLAP = 0.5

cfg.RPN_ENABLE_OHEM = 1
cfg.FACE_LANDMARK = True

cfg.USE_BLUR = False
cfg.USE_OCCLUSION = False




class MultiTaskLoss(nnn.Module):
    def __init__(self, num_classes=2):
        super(MultiTaskLoss, self).__init__()
        anchor = 1
        self.num_classes = num_classes   # include background


    def forward(self, predictions, anchors, targets):
        if cfg.FACE_LANDMARK:
            # conf_data, loc_data, landmark_data = predictions
            conf_pred_batch, loc_pred_batch, landmark_pred_batch = predictions
        else:
            # conf_data, loc_data = predictions
            conf_pred_batch, loc_pred_batch = predictions
        batch = conf_pred_batch.size(0)

        
        anchors_label_list = list()
        bbox_targets_list = list()
        bbox_weights_list = list()
        landmark_targets_list = list()
        landmark_weights_list = list()

        for idx in range(batch):
            anchors_cp = copy.deepcopy(anchors)
            gt_boxes = targets[idx][0] # TODO
            if cfg.FACE_LANDMARK:
                gt_landmarks = targets[idx][1]     

            # TODO
            # num = anchors_cp.size(0)  # total anchors

            # assert anchors_cp.size(1) == conf_pred.size(1)
            # anchors_cp = anchors_cp[: conf_pred.size(1), :]
            # anchors_cp = anchors_cp.size(0)
            # num_classes = self.num_classes

            overlaps = box_overlaps(anchors_cp, gt_boxes)  # support box 5 point
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(anchors_cp.shape[0]), argmax_overlaps]

            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
            # gt_argmax_overlaps = np.where(gt_max_overlaps == overlaps)[0]
            anchors_label = np.empty((anchors_cp.shape[0],), dtype=np.float32)
            anchors_label.fill(-1)

            # assign bg labels first so that positive labels can clobber them
            anchors_label[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0
            # fg label: above threshold IoU
            anchors_label[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1

            if cfg.RPN_ENABLE_OHEM == 0:
                # TODO OHEM MY TODO
                fg_idx = np.where(anchors_label == 1)[0]
                fg_nums = len(fg_idx)

                bg_idx = np.where(anchors_label == 0)[0]
                bg_nums = len(bg_idx)
                if bg_nums > fg_nums*3:
                    disable_idx = npr.choice(fg_idx, size=bg_nums-fg_nums*3, replace=False)
                    anchors_label[disable_idx] = -1
            elif cfg.RPN_ENABLE_OHEM == 1:
                # Reference official
                fg_idx = np.where(anchors_label == 1)[0]
                cfg.RPN_FG_FRACTION = 0.25
                cfg.RPN_BATCH_SIZE = 256
                fg_nums = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCH_SIZE)
                if len(fg_idx) > fg_nums:
                    disable_idx = npr.choice(fg_idx, size=(len(fg_idx) - fg_nums), replace=False)
                    anchors_label[disable_idx] = -1
                
                # subsample negative labels if we have too many
                bg_nums = cfg.RPN_BATCH_SIZE - np.sum(anchors_label == 1)
                bg_idx = np.where(anchors_label == 0)[0]
                if len(bg_idx) > bg_nums:
                    disable_idx = npr.choice(bg_idx, size=(len(bg_idx) - bg_nums), replace=False)
                    anchors_label[disable_idx] = -1
            else:
                fg_idx = np.where(anchors_label == 1)[0]
                fg_nums = len(fg_idx)
                bg_idx = np.where(anchors_label == 0)[0]
                bg_nums = len(bg_idx)
            
            bbox_pred_len = 4
            if cfg.USE_BLUR:
                bbox_pred_len = 5
            else:
                bbox_pred_len = 4
            bbox_targets = np.zeros((anchors_cp.shape[0], bbox_pred_len), dtype=np.float32)
            bbox_targets[:,:] = bbox_transform(anchors_cp, gt_boxes[argmax_overlaps, :])
            
            bbox_weights = np.zeros((anchors_cp.shape[0], bbox_pred_len), dtype=np.float32)
            bbox_weights[anchors_label == 1, 0:4] = 1.0
            if bbox_pred_len>4:
                bbox_weights[anchors_label == 1, 4: bbox_pred_len] = 0.1
            
            if cfg.FACE_LANDMARK:
                if cfg.USE_OCCLUSION:
                    landmark_pred_len = 15
                else:
                    landmark_pred_len = 10

                landmark_targets = np.zeros((anchors_cp.shape[0], landmark_pred_len), dtype=np.float32)
                landmark_weights = np.zeros((anchors_cp.shape[0], landmark_pred_len), dtype=np.float32)
                if landmark_pred_len==10:
                    landmark_weights[anchors_label == 1, :] = 1.0
                elif landmark_pred_len==15:
                    v = [1.0, 1.0, 0.1] * 5
                    assert len(v)==15
                    landmark_weights[anchors_label == 1, :] = np.array(v)
                else:
                    assert False
                #TODO here
                if gt_landmarks.size > 0:
                    a_landmarks = gt_landmarks[argmax_overlaps,:,:]
                    landmark_targets[:] = landmark_transform(anchors_cp, a_landmarks)
                    invalid = np.where(a_landmarks[:, 0, 2]<0.0)[0]  # 
                    landmark_weights[invalid, :] = 0.0

            # anchors_label
            # bbox_targets
            # bbox_weights
            # landmark_targets
            # landmark_weights
            anchors_label_list.append(anchors_label)
            bbox_targets_list.append(bbox_targets)
            bbox_weights_list.append(bbox_weights)
            if cfg.FACE_LANDMARK:
                landmark_targets_list.append(landmark_targets)
                landmark_weights_list.append(landmark_weights)
            
        anchors_label_t = torch.LongTensor(anchors_label_list).cuda().view(batch, -1)
        if cfg.USE_BLUR:
            bbox_pred_len = 5
        else:
            bbox_pred_len = 4
        bbox_targets_t = torch.Tensor(bbox_targets_list).cuda().view(batch, -1, bbox_pred_len)
        bbox_weights_t = torch.Tensor(bbox_weights_list).cuda().view(batch, -1, bbox_pred_len)

        if cfg.FACE_LANDMARK:
            if cfg.USE_OCCLUSION:
                landmark_pred_len = 15
            else:
                landmark_pred_len = 10
            landmark_targets_t = torch.Tensor(landmark_targets_list).cuda().view(batch, -1, landmark_pred_len)
            landmark_weights_t = torch.Tensor(landmark_weights_list).cuda().view(batch, -1, landmark_pred_len) 

        # if cfg.RPN_ENABLE_OHEM>=2:
        #     pass # TODO
        # else:
        #     pass
        # loss_conf = list()
        # loss_loc = list()
        # if cfg.FACE_LANDMARK:
        #     loss_landmark = list() 

        loss_conf = list()
        loss_loc = list()
        loss_landmark = list()

        valid_idx = anchors_label>0
        inds = [0, 12800, 12800+3200, 16800]
        for i in range(len(inds)-1):
            v_idx = valid_idx[inds[i]: inds[i+1]]
            conf_pred = conf_pred_batch[:, v_idx, :].view(-1, self.num_classes)
            conf_t = anchors_label_t[:, v_idx].view(-1,)
            
            loc_pred = loc_pred_batch[:, v_idx, :].view(-1, bbox_pred_len)
            loc_t = bbox_targets_t[:, v_idx, :].view(-1, bbox_pred_len)
            bbox_weights_temp = bbox_weights_t[:, v_idx, :].view(-1, bbox_pred_len)

            if cfg.FACE_LANDMARK:
                if cfg.USE_OCCLUSION:
                    landmark_pred_len = 15
                else:
                    landmark_pred_len = 10
                landmark_pred = landmark_targets_t[:, v_idx, :].view(-1, landmark_pred_len)
                landmark_t = landmark_pred_batch[:, v_idx, :].view(-1, landmark_pred_len)
                landmark_weight_temp = landmark_weights_t[:, v_idx, :].view(-1, landmark_pred_len)

            # conf_ignore_idx = (conf_t == -1).view(-1, 1)
            # compute loss
            # cls
            N_cls = (conf_t!=-1).sum()   # TODO N_cls = (conf_t!=-1).data.sum()
            # loss_conf.append(F.cross_entropy(conf_pred, conf_t, ignore_index=-1, reduction='mean')/float(N_cls))
            loss_conf.append(F.cross_entropy(conf_pred, conf_t, ignore_index=-1, reduction='sum')/float(N_cls))
            
            # box
            loc_pred = loc_pred * bbox_weights_temp
            loc_t = loc_t * bbox_weights_temp
            N_loc = (conf_t == 1).sum() 
            loss_loc.append(F.smooth_l1_loss(loc_pred, loc_t, size_average=False) / float(N_loc))

            # landmark
            landmark_pred *= landmark_weight_temp
            landmark_t *= landmark_weight_temp
            N_landmark = (landmark_weight_temp[:, 0]==1).sum()
            loss_landmark.append(F.smooth_l1_loss(landmark_pred, landmark_t, size_average=False) / float(N_landmark))
            print("N_cls: ", N_cls)
            print("N_loc: ", N_loc)
            print("N_landmark: ", N_landmark)
        return loss_conf, loss_loc, loss_landmark


    def forward_old(self, predictions, anchors, targets):
        if cfg.FACE_LANDMARK:
            # conf_data, loc_data, landmark_data = predictions
            conf_pred, loc_pred, landmark_pred = predictions
        else:
            # conf_data, loc_data = predictions
            conf_pred, loc_pred = predictions

        gt_boxes = targets[0] # TODO
        if cfg.FACE_LANDMARK:
            gt_landmarks = targets[1]
        # TODO
        num = conf_pred.size(0)
        assert anchors.size(0) == conf_pred.size(1)
        anchors = anchors[: conf_pred.size(1), :]
        num_anchors = anchors.size(0)
        num_classes = self.num_classes

        overlaps = box_overlaps(anchors, gt_boxes)
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(anchors.shape[0]), argmax_overlaps]

        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        # gt_argmax_overlaps = np.where(gt_max_overlaps == overlaps)[0]
        anchors_label = np.empty((anchors.shape[0],), dtype=np.float32)
        anchors_label.fill(-1)

        # assign bg labels first so that positive labels can clobber them
        anchors_label[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0
        # fg label: above threshold IoU
        anchors_label[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1

        
        
        if cfg.RPN_ENABLE_OHEM == 0:
            # TODO OHEM MY TODO
            fg_idx = np.where(anchors_label == 1)[0]
            fg_nums = len(fg_idx)

            bg_idx = np.where(anchors_label == 0)[0]
            bg_nums = len(bg_idx)
            if bg_nums > fg_nums*3:
                disable_idx = npr.choice(fg_idx, size=bg_nums-fg_nums*3, replace=False)
                anchors_label[disable_idx] = -1
        elif cfg.RPN_ENABLE_OHEM == 1:
            # Reference official
            fg_idx = np.where(anchors_label == 1)[0]
            cfg.RPN_FG_FRACTION = 0.25
            cfg.RPN_BATCH_SIZE = 256
            fg_nums = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCH_SIZE)
            if len(fg_idx) > fg_nums:
                disable_idx = npr.choice(fg_idx, size=(len(fg_idx) - fg_nums), replace=False)
                anchors_label[disable_idx] = -1
            
            # subsample negative labels if we have too many
            bg_nums = cfg.RPN_BATCH_SIZE - np.sum(anchors_label == 1)
            bg_idx = np.where(anchors_label == 0)[0]
            if len(bg_idx) > bg_nums:
                disable_idx = npr.choice(bg_idx, size=(len(bg_idx) - bg_nums), replace=False)
                anchors_label[disable_idx] = -1
        else:
            fg_idx = np.where(anchors_label == 1)[0]
            fg_nums = len(fg_idx)
            bg_idx = np.where(anchors_label == 0)[0]
            bg_nums = len(bg_idx)
        
        bbox_pred_len = 4
        if cfg.USE_BLUR:
            bbox_pred_len = 5
        else:
            bbox_pred_len = 4
        bbox_targets = np.zeros((anchors.shape[0], 4), dtype=np.float32)
        bbox_targets[:,:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])
        
        bbox_weights = np.zeros((anchors.shape[0], bbox_pred_len), dtype=np.float32)
        bbox_weights[anchors_label == 1, 0:4] = 1.0
        if bbox_pred_len>4:
            bbox_weights[anchors_label == 1, 4: bbox_pred_len] = 0.1
        
        if cfg.FACE_LANDMARK:
            if cfg.USE_OCCLUSION:
                landmark_pred_len = 15
            else:
                landmark_pred_len = 10

            landmark_targets = np.zeros((anchors.shape[0], landmark_pred_len), dtype=np.float32)
            landmark_weights = np.zeros((anchors.shape[0], landmark_pred_len), dtype=np.float32)
            if landmark_pred_len==10:
                landmark_weights[anchors_label == 1, :] = 1.0
            elif landmark_pred_len==15:
                v = [1.0, 1.0, 0.1] * 5
                assert len(v)==15
                landmark_weights[anchors_label == 1, :] = np.array(v)
            else:
                assert False
            #TODO here
            if gt_landmarks.size > 0:
                a_landmarks = gt_landmarks[argmax_overlaps,:,:]
                landmark_targets[:] = landmark_transform(anchors, a_landmarks)
                invalid = np.where(a_landmarks[:, 0, 2]<0.0)[0]  # 
                landmark_weights[invalid, :] = 0.0

        # anchors_label
        # bbox_targets
        # bbox_weights
        # landmark_targets
        # landmark_weights

        # convert to gpu tensor
        anchors_label = torch.LongTensor(anchors_label).cuda()#.view(-1, 1)
        bbox_targets = torch.Tensor(bbox_targets).cuda().view(-1, 4)
        if cfg.FACE_LANDMARK:
            landmark_targets = torch.Tensor(landmark_targets).cuda().view(-1, 10)
        

        if cfg.RPN_ENABLE_OHEM>=2:
            pass # TODO
        else:
            pass
        loss_conf = list()
        loss_loc = list()
        if cfg.FACE_LANDMARK:
            loss_landmark = list() 

        valid_idx = anchors_label>0
        inds = [0, 80*80*2, 80*80*2+40*40*2, 80*80*2+40*40*2+20*20*2]
        for i in range(len(inds)-1):
            idx = valid_idx[inds[i]: inds[i+1]]



        # conf_data, loc_data, landmark_data = predictions

        #match anchors (default boxes) and ground truth boxes
        conf_t = torch.Tensor(num, num_anchors)
        loc_t = torch.Tensor(num, num_anchors, 4)
        landmark_t = torch.Tensor(num, num_anchors, 10)
        
            

    

        

    
