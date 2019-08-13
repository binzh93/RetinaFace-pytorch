import torch
import torch.nn.functional as F

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