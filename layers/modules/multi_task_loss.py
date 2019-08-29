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
            # total_v = 0
            # print("aaa")
            # for k, v in enumerate(targets[0]):
            #     print(v.size(), targets[1][k].size())
                
            #     total_v += v.shape[0]
            # print("landmark gt nums: ", total_v)
            # print(targets[1])
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
            
            # print(idx)
            gt_boxes = targets[0][idx][:,0: -1].detach() # TODO
            gt_labels = targets[0][idx][:, -1].detach()
            # print(gt_boxes)
            # print(gt_labels.shape)
            if cfg.FACE_LANDMARK:
                gt_landmarks = targets[1][idx].detach()   
            # print(gt_boxes)
            # print(gt_landmarks)
#             conf_pred_batch[idx] = conf_pred_batch[idx].view(-1, self.num_classes)
#             loc_pred_batch[idx] = loc_pred_batch[idx].view(-1, 4)
#             if cfg.FACE_LANDMARK:
#                 landmark_pred_batch[idx] = landmark_pred_batch[idx].view(-1, 10)
            
            # overlaps = box_overlaps(anchors.detach(), gt_boxes)  # not support box 5 point
            overlaps = box_overlaps(gt_boxes, anchors.detach())   # (gt_boxes_nums, anchors_num)

            best_gt_overlap_per_anchor, best_gt_idx_per_anchor = overlaps.max(0, keepdim=True)
            # TODO
            # best_anchor_overlap_per_gt, best_anchor_idx_per_gt = overlaps.max(1, keepdim=True)  
            
            best_gt_overlap_per_anchor.squeeze_(0)
            best_gt_idx_per_anchor.squeeze_(0)

            anchor_conf_t = gt_labels[best_gt_idx_per_anchor]    
            # anchor_conf_t[best_gt_overlap_per_anchor >= cfg.RPN_POSITIVE_OVERLAP]
            anchor_conf_t[best_gt_overlap_per_anchor < cfg.RPN_POSITIVE_OVERLAP] = -1  # fill -1 except the pos 
            # print(anchor_conf_t[anchor_conf_t>0].sum())
            if cfg.RPN_ENABLE_OHEM == 1:
                pos_idx = best_gt_overlap_per_anchor>=cfg.RPN_POSITIVE_OVERLAP            
                neg_all_idx = best_gt_overlap_per_anchor<cfg.RPN_NEGATIVE_OVERLAP
                assert neg_all_idx.sum() >= (3 * pos_idx.sum())
                neg_argsort = torch.argsort(best_gt_overlap_per_anchor, 0)
                neg_nums = 3 * pos_idx.sum()
                neg_count = 0
                for k, v in enumerate(neg_argsort):
                    if neg_count >= neg_nums:
                        break
                    if pos_idx[v]:
                        print("Sorry, you should change your anchor settings, some gt_box can't find fitted anchor")
                        print(v, best_gt_idx_per_anchor[0][v], best_gt_overlap_per_anchor[v])
                        continue
                    anchor_conf_t[v] = 0  # set as bg
                    neg_count += 1
                # print(time.time() - tt1)
            # import pdb
            # pdb.set_trace()
            # pos_idx = anchor_conf_t > 0
            bbox_targets = bbox_transform2(anchors.detach(), gt_boxes[best_gt_idx_per_anchor, :])
            if cfg.FACE_LANDMARK:
                # print(gt_landmarks.shape)
                landmark_target = landmark_transform2(anchors.detach(), gt_landmarks[best_gt_idx_per_anchor, :, :]).view(-1, 10)

            anchor_label_batch[idx] = anchor_conf_t
            loc_t_batch[idx] = bbox_targets
            loc_weights[idx][pos_idx] = 1.0  # only pos 
            
            # import pdb
            # pdb.set_trace()
            if cfg.FACE_LANDMARK:
                landmark_t_batch[idx] = landmark_target
                landmark_weights[idx][pos_idx] = 1.0  # get all pos idx
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

        # print(anchor_conf[anchor_conf == -1].sum()) # other
        # print(anchor_conf[anchor_conf == 0].sum())  # neg
        # print(anchor_conf[anchor_conf > 0].sum())   # pos

        # tt1 = time.time()
        # loc_t = torch.Tensor(batch, 16800, 4).cuda()
        # conf_t = torch.LongTensor(batch, 16800).cuda()
        # print("1: ", time.time() - tt1)

            
           
        # import pdb
        # pdb.set_trace()
    
        loss_conf = list()
        loss_loc = list()
        if cfg.FACE_LANDMARK:
            loss_landmark = list()
        inds = [0, 12800, 12800+3200, 16800]
        # print(landmark_weights.sum())  # TODO dasdasdsad dasdsadsadsa
        for i in range(len(inds)-1):
            # TODO if need to contiguous
            # import pdb
            # pdb.set_trace()

            # cls loss
            conf_pred_batch_feat = conf_pred_batch[i].view(-1, self.num_classes)
            # import pdb
            # pdb.set_trace()
            # print(conf_pred_batch[i].size(), conf_pred_batch_feat.size())
            # print(batch*(inds[i+1]-inds[i]), conf_pred_batch_feat.size(0))
            assert (batch*(inds[i+1]-inds[i])) == conf_pred_batch_feat.size(0)
            anchor_label_batch_feat = anchor_label_batch[:, inds[i]: inds[i+1]].contiguous().view(conf_pred_batch_feat.size(0), )
            N_cls = (anchor_label_batch_feat != -1).sum()   # if detach must
            # print(anchor_label_batch_feat.size(), N_cls)
            # print(N_cls)
            # loss_conf.append(F.cross_entropy(conf_pred_batch_feat, anchor_label_batch_feat, ignore_index=-1, reduction='sum')/float(N_cls))
            cls_feat_loss = F.cross_entropy(conf_pred_batch_feat, anchor_label_batch_feat, ignore_index=-1, reduction='sum')
            if N_cls>0:
                cls_feat_loss = cls_feat_loss / float(N_cls)
            loss_conf.append(cls_feat_loss)

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



    def forward_old_n1(self, predictions, anchors, targets):
        if cfg.FACE_LANDMARK:
            # conf_data, loc_data, landmark_data = predictions
            conf_pred_batch, loc_pred_batch, landmark_pred_batch = predictions
        else:
            # conf_data, loc_data = predictions
            conf_pred_batch, loc_pred_batch = predictions
#         batch = conf_pred_batch.size(0)
        batch = conf_pred_batch[0].size(0)

        anchors_label_list = list()
        bbox_targets_list = list()
        bbox_weights_list = list()
        landmark_targets_list = list()
        landmark_weights_list = list()

        for idx in range(batch):
            anchors_cp = copy.deepcopy(anchors)
            gt_boxes = targets[0][idx] # TODO
            if cfg.FACE_LANDMARK:
                gt_landmarks = targets[1][idx]     

            # TODO
            # num = anchors_cp.size(0)  # total anchors

            # assert anchors_cp.size(1) == conf_pred.size(1)
            # anchors_cp = anchors_cp[: conf_pred.size(1), :]
            # anchors_cp = anchors_cp.size(0)
            # num_classes = self.num_classes

            overlaps = box_overlaps(anchors_cp, gt_boxes)  # support box 5 point
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(anchors_cp.shape[0]), argmax_overlaps]
            
            # print("==========")
            # print(gt_boxes.shape, gt_landmarks.shape)
            # print(argmax_overlaps)

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
            print(type(anchors_cp), type(gt_boxes))
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
                    print(gt_landmarks.shape)
                    a_landmarks = gt_landmarks[argmax_overlaps,:]
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

        valid_idx = anchors_label_t>0
        print(valid_idx)
        inds = [0, 12800, 12800+3200, 16800]
        for i in range(len(inds)-1):
            v_idx = valid_idx[inds[i]: inds[i+1]]
#             print(batch, conf_pred_batch[i].size())
            conf_pred = conf_pred_batch[i].view(-1, self.num_classes)
#             print("anchors_label_t: ", anchors_label_t.shape)
#             conf_t = anchors_label_t[:, v_idx].view(-1,)
            conf_t = anchors_label_t[v_idx].view(-1,)
            
#             loc_pred = loc_pred_batch[:, v_idx, :].view(-1, bbox_pred_len)
            loc_pred = loc_pred_batch[i].view(-1, bbox_pred_len)
#             loc_t = bbox_targets_t[:, v_idx, :].view(-1, bbox_pred_len)
            pos_loc_idx = valid_idx.unsqueeze(valid_idx.dim()).expand_as(bbox_targets_t)
            loc_t = bbox_targets_t[v_idx].view(-1, bbox_pred_len)
            bbox_weights_temp = bbox_weights_t[:, v_idx, :].view(-1, bbox_pred_len)

            if cfg.FACE_LANDMARK:
                if cfg.USE_OCCLUSION:
                    landmark_pred_len = 15
                else:
                    landmark_pred_len = 10
#                 landmark_pred = landmark_targets_t[:, v_idx, :].view(-1, landmark_pred_len)
                landmark_pred = landmark_targets_t[i].view(-1, landmark_pred_len)
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
        
            

    

        

    
