import torch
import torch.nn as nn
from utils.box_utils import box_overlaps, bbox_transform, landmark_transform, bbox_transform2, landmark_transform2
import numpy as np
import numpy.random as npr
import copy
import torch.nn.functional as F
import time


from easydict import EasyDict as edict
__C = edict()
cfg = __C
cfg.RPN_NEGATIVE_OVERLAP = 0.3
cfg.RPN_POSITIVE_OVERLAP = 0.5

cfg.RPN_ENABLE_OHEM = 1
# cfg.FACE_LANDMARK = True
cfg.FACE_LANDMARK = False


cfg.USE_BLUR = False
cfg.USE_OCCLUSION = False
cfg.COPY_MXNET = False



class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiTaskLoss, self).__init__()
        anchor = 1
        self.num_classes = num_classes   # include background
        self.negpos_ratio = 3.0
        
    def forward(self, predictions, anchors, targets):
        '''
        Args:
            conf_pred_batch: [(batch_size, 0: 12800, num_classes), 
                              (batch_size, 12800: 12800+3200, num_classes),
                              (batch_size, 12800+3200: num_priors, num_classes)]
            loc_pred_batch: [(batch_size, 0: 12800, 4), 
                             (batch_size, 12800: 12800+3200, 4),
                             (batch_size, 12800+3200: num_priors, 4)]
            landmark_pred_batch: [(batch_size, 0: 12800, 10), 
                                  (batch_size, 12800: 12800+3200, 10),
                                  (batch_size, 12800+3200: num_priors, 10)]
            anchors: (num_anchors, 4)   (x_min, y_min, x_max, y_max)
            landmark_t: (num_anchors, 5, 2) or (num_anchors, 5, 3)

        '''

        t1 = time.time()
        if cfg.FACE_LANDMARK:
            # conf_data, loc_data, landmark_data = predictions
            conf_pred_batch, loc_pred_batch, landmark_pred_batch = predictions
        else:
            # conf_data, loc_data = predictions
            conf_pred_batch, loc_pred_batch = predictions
            # conf_pred_batch = predictions

#         batch = conf_pred_batch.size(0)
        batch = conf_pred_batch[0].size(0)

        anchor_label_batch = torch.LongTensor(batch, anchors.size(0))
        loc_t_batch = torch.Tensor(batch, anchors.size(0), 4)
        loc_weights = torch.zeros((batch, anchors.size(0), 4), dtype=torch.float32)
        if cfg.FACE_LANDMARK:
            landmark_t_batch = torch.Tensor(batch, anchors.size(0), 10)
            landmark_weights = torch.zeros((batch, anchors.size(0), 10), dtype=torch.float32)
        
        t2 = time.time()  # something load

        for idx in range(batch): 
            gt_boxes = targets[0][idx][:,0: -1].detach() # TODO
            gt_labels = targets[0][idx][:, -1].detach()
            if cfg.FACE_LANDMARK:
                gt_landmarks = targets[1][idx].detach()   
            
            # overlaps = box_overlaps(anchors.detach(), gt_boxes)  # not support box 5 point
            overlaps = box_overlaps(gt_boxes, anchors.detach())   # (gt_boxes_nums, anchors_num)

            best_gt_overlap_per_anchor, best_gt_idx_per_anchor = overlaps.max(0, keepdim=True)
            # TODO
            # best_anchor_overlap_per_gt, best_anchor_idx_per_gt = overlaps.max(1, keepdim=True)  
            
            best_gt_overlap_per_anchor.squeeze_(0)
            best_gt_idx_per_anchor.squeeze_(0)

            anchor_conf_t = gt_labels[best_gt_idx_per_anchor]    
            # anchor_conf_t[best_gt_overlap_per_anchor >= cfg.RPN_POSITIVE_OVERLAP]
            anchor_conf_t[best_gt_overlap_per_anchor < cfg.RPN_POSITIVE_OVERLAP] = 0  # fill 0 except the pos 
 
            # pos_idx = anchor_conf_t > 0

            bbox_targets = bbox_transform2(anchors.detach(), gt_boxes[best_gt_idx_per_anchor, :])
            if cfg.FACE_LANDMARK:
                # print(gt_landmarks.shape)
                landmark_target = landmark_transform2(anchors.detach(), gt_landmarks[best_gt_idx_per_anchor, :, :]).view(-1, 10)

            anchor_label_batch[idx] = anchor_conf_t
            loc_t_batch[idx] = bbox_targets
            loc_weights[idx][anchor_conf_t > 0] = 1.0  # only pos 

            if cfg.FACE_LANDMARK:
                landmark_t_batch[idx] = landmark_target
                landmark_weights[idx][anchor_conf_t > 0] = 1.0  # get all pos idx
                invalid_landmark_idx = gt_landmarks[best_gt_idx_per_anchor][:, 0, 2] < 0
                landmark_weights[idx][invalid_landmark_idx] = 0.0  # filter by -1 from pos
                # valid_landmark_idx = gt_landmarks[best_gt_idx_per_anchor][:, 0, 2] >= 0
                # landmark_weights[idx][valid_landmark_idx] = 1.0  # filter by -1 from pos
        t3 = time.time()  # batch compute overlaps

        anchor_label_batch = anchor_label_batch.cuda()
        loc_t_batch = loc_t_batch.cuda()
        loc_weights = loc_weights.cuda()
        if cfg.FACE_LANDMARK:
            landmark_t_batch = landmark_t_batch.cuda()
            landmark_weights = landmark_weights.cuda()

        # import pdb
        # pdb.set_trace()
        # OHEM
        if cfg.RPN_ENABLE_OHEM == 2:
            inds = [0, 12800, 12800+3200, 16800]
            for i in range(len(inds)-1):
                # Compute max conf across batch for hard negative mining
                batch_conf_feat = conf_pred_batch[i].view(-1,self.num_classes)[batch* inds[i]: batch* inds[i+1], :].clone()
                anchor_label_batch_feat = anchor_label_batch[:, inds[i]: inds[i+1]].clone()
                loss_c = log_sum_exp(batch_conf_feat) - batch_conf_feat.gather(1, anchor_label_batch_feat.view(-1, 1))
                # loss_c = log_sum_exp(batch_conf_feat)

                pos = anchor_label_batch_feat>0

                # Hard Negative Mining
                loss_c[pos.view(-1,1)] = 0 # filter out pos boxes for now
                loss_c = loss_c.view(batch, -1)
                _,loss_idx = loss_c.sort(1, descending=True)
                _,idx_rank = loss_idx.sort(1)
                num_pos = pos.long().sum(1, keepdim=True)

                if num_pos.data.sum() > 0:
                    num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
                else:
                    print("multi task loss ")
                    raise NotImplementedError
                    # fake_num_pos = torch.ones(32, 1).long() * 15
                    # num_neg = torch.clamp(self.negpos_ratio * fake_num_pos, max=pos.size(1) - 1)
                neg = idx_rank < num_neg.expand_as(idx_rank)
                
                # Confidence Loss Including Positive and Negative Examples
                pos_idx = pos.unsqueeze(2).expand_as(conf_data)
                neg_idx = neg.unsqueeze(2).expand_as(conf_data)
                conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
                targets_weighted = conf_t[(pos+neg).gt(0)]
        

        # print(anchor_conf[anchor_conf == -1].sum()) # other
        # print(anchor_conf[anchor_conf == 0].sum())  # neg
        # print(anchor_conf[anchor_conf > 0].sum())   # pos
 
        loss_conf = list()
        loss_loc = list()
        if cfg.FACE_LANDMARK:
            loss_landmark = list()
        inds = [0, 12800, 12800+3200, 16800]
        # print(landmark_weights.sum())  # TODO dasdasdsad dasdsadsadsa
        
        for i in range(len(inds)-1):
            if cfg.RPN_ENABLE_OHEM == 1:
                # Compute max conf across batch for hard negative mining
                conf_pred_batch_feat = conf_pred_batch[i].view(-1,self.num_classes).contiguous()#.clone()
                anchor_label_batch_feat = anchor_label_batch[:, inds[i]: inds[i+1]].contiguous()#.clone()
                # print(conf_pred_batch_feat.shape, anchor_label_batch_feat.shape)
                # import pdb
                # pdb.set_trace()
                # loss_c = log_sum_exp(conf_pred_batch_feat) - conf_pred_batch_feat.gather(1, anchor_label_batch_feat.view(-1, 1))
                loss_c = log_sum_exp2(conf_pred_batch_feat) - conf_pred_batch_feat.gather(1, anchor_label_batch_feat.view(-1, 1))

                pos = anchor_label_batch_feat>0

                # Hard Negative Mining
                loss_c[pos.view(-1,1)] = 0 # filter out pos boxes for now
                loss_c = loss_c.view(batch, -1)
                _,loss_idx = loss_c.sort(1, descending=True)
                _,idx_rank = loss_idx.sort(1)
                num_pos = pos.long().sum(1, keepdim=True)

                # print("num_pos: ", pos.size(), num_pos.size())

                if num_pos.data.sum() > 0:
                    num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1)-1)
                else:
                    print("multi task loss: use fake num ")
                    fake_num_pos = torch.ones(batch, 1).long().cuda() * 15
                    num_neg = torch.clamp(self.negpos_ratio * fake_num_pos, max=pos.size(1) - 1)

                neg = idx_rank < num_neg.expand_as(idx_rank)
                
                # Confidence Loss Including Positive and Negative Examples

                pos_idx = pos.view(batch*(inds[i+1]-inds[i]), ).unsqueeze(1).expand_as(conf_pred_batch_feat)
                neg_idx = neg.view(batch*(inds[i+1]-inds[i]), ).unsqueeze(1).expand_as(conf_pred_batch_feat)
                # conf_p = conf_pred_batch_feat[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
                # targets_weighted = conf_pred_batch_feat[(pos+neg).gt(0)]
            else:
                raise NotImplementedError
            # import pdb
            # pdb.set_trace()
            # cls loss
            conf_pred_batch_feat_valid = conf_pred_batch_feat[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            anchor_label_batch_feat_valid = anchor_label_batch_feat[(pos+neg).gt(0)]
            if anchor_label_batch_feat_valid.size()[0] > 0:
                cls_feat_loss = F.cross_entropy(conf_pred_batch_feat_valid, anchor_label_batch_feat_valid, reduction='mean')
                loss_conf.append(cls_feat_loss)
                # print(i, conf_pred_batch_feat_valid.size(), anchor_label_batch_feat_valid.size())
            else:
                print("No bg and fg")
                import pdb
                pdb.set_trace()
                raise NotImplementedError

            # # cls loss
            # conf_pred_batch_feat = conf_pred_batch[i].view(-1, self.num_classes)
            # # print(conf_pred_batch[i].size(), conf_pred_batch_feat.size())
            # # print(batch*(inds[i+1]-inds[i]), conf_pred_batch_feat.size(0))
            # assert (batch*(inds[i+1]-inds[i])) == conf_pred_batch_feat.size(0)
            # anchor_label_batch_feat = anchor_label_batch[:, inds[i]: inds[i+1]].contiguous().view(conf_pred_batch_feat.size(0), )
            # N_cls = (anchor_label_batch_feat != -1).sum()   # if detach must

            # # loss_conf.append(F.cross_entropy(conf_pred_batch_feat, anchor_label_batch_feat, ignore_index=-1, reduction='sum')/float(N_cls))
            # cls_feat_loss = F.cross_entropy(conf_pred_batch_feat, anchor_label_batch_feat, ignore_index=-1, reduction='sum')

            # if N_cls>0:
            #     cls_feat_loss = cls_feat_loss / float(N_cls)
            # loss_conf.append(cls_feat_loss)




            # loc loss
            loc_pred_batch_feat = loc_pred_batch[i].view(-1, 4)
            loc_t_batch_feat = loc_t_batch[:, inds[i]: inds[i+1]].contiguous().view(-1, 4)
            loc_weights_feat = loc_weights[:, inds[i]: inds[i+1]].contiguous().view(-1, 4)

            loc_pred_batch_feat = loc_pred_batch_feat * loc_weights_feat
            loc_t_batch_feat = loc_t_batch_feat * loc_weights_feat
            N_loc = loc_weights_feat.detach().sum() / 4.0  # if detach must
            # print(N_loc, batch)
            if cfg.COPY_MXNET:
                loss_loc.append(F.smooth_l1_loss(loc_pred_batch_feat, loc_t_batch_feat, reduction='sum') / float(batch))
            else:
                loc_feat_loss = F.smooth_l1_loss(loc_pred_batch_feat, loc_t_batch_feat, reduction='sum') 
                if N_loc>0:
                    loc_feat_loss = loc_feat_loss / float(N_loc)
                loss_loc.append(loc_feat_loss)

            # landmark loss
            if cfg.FACE_LANDMARK:
                landmark_pred_batch_feat = landmark_pred_batch[i].view(-1, 10)
                landmark_t_batch_feat = landmark_t_batch[:, inds[i]: inds[i+1]].contiguous().view(-1, 10)
                landmark_weights_feat = landmark_weights[:, inds[i]: inds[i+1]].contiguous().view(-1, 10)

                landmark_pred_batch_feat = landmark_pred_batch_feat * landmark_weights_feat
                landmark_t_batch_feat = landmark_t_batch_feat * landmark_weights_feat
                N_landmark = landmark_weights_feat.detach().sum() / 10.0 # if detach must
                # print(N_landmark, batch)
                if cfg.COPY_MXNET:
                    loss_landmark.append(F.smooth_l1_loss(landmark_pred_batch_feat, landmark_t_batch_feat, reduction='sum') * 0.4 / float(batch))
                else:
                    landmark_feat_loss = F.smooth_l1_loss(landmark_pred_batch_feat, landmark_t_batch_feat, reduction='sum') * 0.4
                    if N_landmark > 0:
                        landmark_feat_loss = landmark_feat_loss / float(N_landmark)
                    loss_landmark.append(landmark_feat_loss)

        t4 = time.time()  # data to gpu and compute loss

        Display = False
        if Display:
            print("Multi task loss ==> something load: ", t2 - t1)
            print("Multi task loss ==> batch compute overlaps: ", t3 - t2)
            print("Multi task loss ==> data to gpu and compute loss: ", t4 - t3)

        
        if cfg.FACE_LANDMARK:
            return loss_conf, loss_loc, loss_landmark
        else:
            return loss_conf, loss_loc
            # return loss_conf



def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def log_sum_exp2(conf_pred_batch_feat):
    # log_cls_loss(conf_pred_batch_feat, anchor_label_batch_feat.view(-1, 1)) 
    
    loss = torch.log(torch.sum(torch.exp(conf_pred_batch_feat), 1, keepdim=True))
    return loss


