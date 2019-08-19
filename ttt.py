import torch
import torch.nn.functional as F
import time

label1 = torch.LongTensor([[1, 0, 1, -1, 0, 0 ,0], [1, 0, 0, 0 ,0, -1, 0]])
# label2 = torch.LongTensor([[1, 0, 1, 1, 0, 0 ,0], [1, 0, 0, 0 ,0, 1, 0]])
# label2 = torch.LongTensor([[1], [1]])
# label2 = torch.LongTensor([[1], [1]])
# gt = torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
gt = torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
# box_weigth = torch.Tensor([[0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3]])

pred = []
pred.append([0.9, 0.9, 0.9, 0.9])
pred.append([0.8, 0.8, 0.8, 0.8])
# pred.append([0.7, 0.7, 0.7, 0.7])
# pred.append([0.6, 0.6, 0.6, 0.6])

# print(label3 < -1)torc
pred = torch.Tensor(pred).view(2, -1, 4)
gt = gt.view(2, -1, 4)
# box_weigth = box_weigth.view(2, -1, 4)
# print(pred.shape)
# print(label2.shape)
# pred = torch.Tensor([[0.9, 0, 1, 0.9, 0, 0 ,0], [1, 0, 0, 0 ,0, 0.9, 0]])

# s = F.cross_entropy(pred.view(3, 2), label3.view(3, )) # , size_average=False
# s = F.cross_entropy(pred.view(3, 2), label3.view(3, ), ignore_index=-1) # , size_average=False
# s = F.cross_entropy(pred.view(3, 2), label3.view(3, ), size_average=False) # , size_average=False

# print(pred)
# print(gt)
# pred *= box_weigth
# gt *= box_weigth
print(pred)
print(gt)
loss_loc = F.smooth_l1_loss(pred, gt, size_average=False)
# loss_loc = F.smooth_l1_loss(pred, gt)
print(loss_loc)


loss_loc = F.smooth_l1_loss(pred, gt, reduction='sum')
print(loss_loc)



a = torch.Tensor([[1, 2], [4, 2], [6, 6]]).view(3, 2)

print(a)
import numpy as np
b = np.array([[1.0, 2.0], [4, 2], [6, 6]], dtype=np.float32).reshape(2, 3)
c = np.array([999, 999], dtype=np.int)
# c = c
d = np.hstack((b, c[:, np.newaxis]))
print(d)
ov, idx = a.max(1, keepdim=True)
x = torch.Tensor([1, 2, 0])
x = x>0
print(x.size())
print(x.unsqueeze(x.dim()).size())
p_idx = x.unsqueeze(x.dim()).expand_as(a)
print(p_idx)
print(a)
print(a[p_idx].shape)
# print(idx)
# print(ov.shape)
# print(idx.shape)


a = np.array([3, 2, 5, 1])
a_t = torch.Tensor(a)
print(a)
idx = torch.Tensor([1, 0, 1, 0])
idx = idx >0
print(idx)
print(a_t[idx])

# labels = np.array([3, 2, 5, 1])
# best_gt_idx_per_anchor.squeeze_(0)
# labels = torch.LongTensor(np.array([3, 2, 5, 1])).cuda()
# conf = labels[best_gt_idx_per_anchor]

tt1 = time.time()
for i in range(1):
    # anchor_label = torch.Tensor(16800, 1).fill_(-1).long().cuda()
    # targets = torch.ones(16800)*-1
    # targets = targets.cuda()
    labels = torch.LongTensor([1, 2, 3, 4, 5])#.cuda()
print(time.time() - tt1)

a = np.array([[1, 2], [3, 4]])
b = torch.Tensor(a)
c = b.detach().copy()
print(b)
print(c)
c[0][0] = 999
print(b)
print(c)


