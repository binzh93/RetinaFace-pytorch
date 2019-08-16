import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time

from models.model_builder import RetinaFace
from models.retina_face import *
from data.face_loader import WiderFaceDetection
from layers.modules import MultiTaskLoss
# from layers.modules import 
from layers.funnctions import Prior_Box

import torch.optim as optim
import time
import argparse

parser = argparse.ArgumentParser(description='RetinaFace')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')

parser.add_argument('-max','--max_epoch', default=100, type=int, help='max epoch for retraining')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--root', default="/home/shanma/Workspace/zhubin/RetinaFace/data/retinaface", help='Dataset root directory path')
parser.add_argument('--dataset_root', default="/home/shanma/Workspace/zhubin/RetinaFace/data/retinaface/train", help='Dataset root directory path')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--frequent', help='frequency of logging', default=30, type=int)
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
args = parser.parse_args()



# def adjust_learning_rate(optimizer, epoch, step_epoch, gamma, epoch_size, iteration):
#     """Sets the learning rate
#     # Adapted from PyTorch Imagenet example:
#     # https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     ## warmup
#     if epoch <= cfg.TRAIN.WARMUP_EPOCH:
#         if cfg.TRAIN.WARMUP:
#             iteration += (epoch_size * (epoch - 1))
#             lr = 1e-6 + (cfg.SOLVER.BASE_LR - 1e-6) * iteration / (
#                 epoch_size * cfg.TRAIN.WARMUP_EPOCH)
#         else:
#             lr = cfg.SOLVER.BASE_LR
#     else:
#         div = 0
#         if epoch > step_epoch[-1]:
#             div = len(step_epoch) - 1
#         else:
#             for idx, v in enumerate(step_epoch):
#                 if epoch > step_epoch[idx] and epoch <= step_epoch[idx + 1]:
#                     div = idx
#                     break
#         lr = cfg.SOLVER.BASE_LR * (gamma**div)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

# def save_checkpoint(net, epoch, size, optimizer):
#     save_name = os.path.join(
#         args.save_folder,
#         cfg.MODEL.TYPE + "_epoch_{}_{}".format(str(epoch), str(size)) + '.pth')
#     torch.save({
#         'epoch': epoch,
#         'size': size,
#         'batch_size': cfg.TRAIN.BATCH_SIZE,
#         'model': net.state_dict(),
#         'optimizer': optimizer.state_dict()
#     }, save_name)

def eval_net(val_dataset, val_loader, net, detector, cfg, transform, max_per_image=300, thresh=0.01, batch_size=1):
    net.eval()


anchors = Prior_Box()

with torch.no_grad():
    anchors = anchors.forward()
    # anchors = anchors.cuda()

def train_net(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma, end_epoch, cfg):
    net.train()
    # print(len(train_loader))
    # print("=======")
    eps = 0.0001
    for iteration in range(end_epoch):
        t_begin = time.time()
        total_loss = 0.0
        loss = 0.0
        loss_32_conf = 0.0
        loss_16_conf = 0.0
        loss_8_conf = 0.0
        
        loss_32_loc = 0.0
        loss_16_loc = 0.0
        loss_8_loc = 0.0
        
        loss_32_landmark = 0.0
        loss_16_landmark = 0.0
        loss_8_landmark = 0.0
        
        t5 = 0.0
        for i, (imgs, boxes, landmarks) in enumerate(train_loader):
            t1 = time.time()
            print("image load: ", t1 - t5)
            imgs = imgs.cuda()

#             with torch.no_grad():
#                 boxes = [anno.cuda() for anno in boxes]
#                 landmarks = [anno.cuda() for anno in landmarks]
            output = net(imgs)
            optimizer.zero_grad()
            
            t2 = time.time() 
            loss_conf, loss_loc, loss_landmark = criterion(output, anchors, [boxes, landmarks])
            t3 = time.time()
            
            loss = loss_conf[0] + loss_conf[1] + loss_conf[2] 
            loss += loss_loc[0] + loss_loc[1] + loss_loc[2] 
            loss += loss_landmark[0] + loss_landmark[1] + loss_landmark[2] 
            
            total_loss += loss
            loss_32_conf += loss_conf[0]
            loss_16_conf += loss_conf[1]
            loss_8_conf += loss_conf[2]

            loss_32_loc += loss_loc[0]
            loss_16_loc += loss_loc[1]
            loss_8_loc += loss_loc[2]

            loss_32_landmark += loss_landmark[0]
            loss_16_landmark += loss_landmark[1]
            loss_8_landmark += loss_landmark[2]
            #print(loss_conf)
            #print("------")
            #print(loss_loc)
            #print("------")
            #print(loss_landmark)
            #print("=======")

            if (i+1) % args.frequent == 0:

                print("Epoch[{}]  Batch [{}-{}]  total loss: {:4.6f}\t"\
                "32_conf: {:4.6f}\t16_conf: {:4.6f}\t8_conf: {:4.6f}\t" \
                "32_loc: {:4.6f}\t16_loc: {:4.6f}\t8_loc: {:4.6f}\t" \
                "32_landmark: {:4.6f}\t16_landmark: {:4.6f}\t8_landmark: {:4.6f}".format(iteration, 0, i+1, total_loss.item()/(i+1), \
                loss_32_conf.item()/(i+1), loss_16_conf.item()/(i+1), loss_8_conf.item()/(i+1), \
                loss_32_loc.item()/(i+1), loss_16_loc.item()/(i+1), loss_8_loc.item()/(i+1), \
                loss_32_landmark.item()/(i+1), loss_16_landmark.item()/(i+1), loss_8_landmark.item()/(i+1) ))
            

            # print("32_conf: {:.6f} 16_conf: {:.6f} 8_conf: {:.6f} 32_loc: {:.6f} 16_loc: {:.6f} 8_loc: {:.6f} 32_landmark: {:.6f} 16_landmark: {:.6f} 8_landmark: {:.6f}".format(loss_conf[0], \
            # loss_conf[1], loss_conf[2], loss_loc[0], loss_loc[1], loss_loc[2], loss_landmark[0], loss_landmark[1], loss_landmark[2]))
#             print("".format(loss_conf[1]), )
#             print("".format(loss_conf[2]), )
#             print("".format(loss_loc[0]), )
#             print("loss_16_loc: {:.6f}".format(loss_loc[1]), )
#             print("loss_8_loc: {:.6f}".format(loss_loc[2]), )
#             print("loss_32_landmark: {:.6f}".format(loss_landmark[0]), )
#             print("loss_16_landmark: {:.6f}".format(loss_landmark[1]), )
#             print("loss_8_landmark: {:.6f}".format(loss_landmark[2]))
            
            #
            t4 = time.time()
            loss.backward()
            optimizer.step()
#             print("time: {}{} total loss: {}{}".format())
#             print()
#             print("loss:", loss.item())
            t5 = time.time()

            print("===========")
            print("image infer: ", t2 - t1)
            print("criterion loss time: ",t3 - t2) 
            print("loss compute: ", t4 - t3)
            print("loss backward: ", t5 - t4)


        t_end = time.time()   
        print("Epoch[{}]  total loss: {:.6f}".format(iteration, total_loss.item()/(i+1)))  # :10.6f
        print("Epoch[{}]  loss_32_conf: {:.6f}".format(iteration, loss_32_conf.item()/(i+1)))
        print("Epoch[{}]  loss_16_conf: {:.6f}".format(iteration, loss_16_conf.item()/(i+1)))
        print("Epoch[{}]  loss_8_conf: {:.6f}".format(iteration, loss_8_conf.item()/(i+1)))
        print("Epoch[{}]  loss_32_loc: {:.6f}".format(iteration, loss_32_loc.item()/(i+1)))
        print("Epoch[{}]  loss_16_loc: {:.6f}".format(iteration, loss_16_loc.item()/(i+1)))
        print("Epoch[{}]  loss_8_loc: {:.6f}".format(iteration, loss_8_loc.item()/(i+1)))
        print("Epoch[{}]  loss_32_landmark: {:.6f}".format(iteration, loss_32_landmark.item()/(i+1)))
        print("Epoch[{}]  loss_16_landmark: {:.6f}".format(iteration, loss_16_landmark.item()/(i+1)))
        print("Epoch[{}]  loss_8_landmark: {:.6f}".format(iteration, loss_8_landmark.item()/(i+1)))
        print("Epoch[{}]  time comsuming: {}".format(iteration, (t_begin-t_end)))     
        # print("=========epoch {}=========".format(iteration+1))
        # print("time comsuming: {}".format((t2-t1)))     
        # print("total loss: {:.6f}".format(loss.item()/(iteration+1)))  # :10.6f
        # print("loss_32_conf: {:.6f}".format(loss_32_conf.item()/(iteration+1)))
        # print("loss_16_conf: {:.6f}".format(loss_16_conf.item()/(iteration+1)))
        # print("loss_8_conf: {:.6f}".format(loss_8_conf.item()/(iteration+1)))
        # print("loss_32_loc: {:.6f}".format(loss_32_loc.item()/(iteration+1)))
        # print("loss_16_loc: {:.6f}".format(loss_16_loc.item()/(iteration+1)))
        # print("loss_8_loc: {:.6f}".format(loss_8_loc.item()/(iteration+1)))
        # print("loss_32_landmark: {:.6f}".format(loss_32_landmark.item()/(iteration+1)))
        # print("loss_16_landmark: {:.6f}".format(loss_16_landmark.item()/(iteration+1)))
        # print("loss_8_landmark: {:.6f}".format(loss_8_landmark.item()/(iteration+1)))


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    
    imgs = []
    targets = []
    landmarks = []
    for sample in batch:
        imgs.append(sample[0])
#         targets.append(torch.FloatTensor(sample[1]))
#         landmarks.append(torch.FloatTensor(sample[2]))
        targets.append(sample[1])
        landmarks.append(sample[2])
    return torch.stack(imgs, 0), targets, landmarks

    
def main():
#     print("=======")
    backbone = resnet50()
    #backbone = resnet18()
    net = RetinaFace(backbone)
    if torch.cuda.is_available():
        if args.cuda:
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if args.num_workers>1:
                net = torch.nn.DataParallel(net)  # must after loading model weigths
            else:
                # raise NotImplementedError
                pass
            net.cuda()
            # net.to(device)
            cudnn.benchmark = True
    
    
            

    train_dataset = WiderFaceDetection(root_path=args.root, data_path=args.dataset_root, phase="train", 
                                       dataset_name="WiderFace", transform=None)
#     train_loader = data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers) # , collate_fn=detection_collate)
    train_loader = data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate)

    print("sucess train_loader")

    
    # priorbox = PriorBox(anchors(cfg))

    # with torch.no_grad():
    #     priors = priorbox.forward()
    #     if cfg.train_cfg.cuda:
    #         priors = priors.cuda()


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    start_epoch = 0
    end_epoch = args.max_epoch
    criterion = MultiTaskLoss()
    for epoch in range(start_epoch + 1, end_epoch + 1):

        ######
        train_net(train_loader, net, criterion, optimizer, epoch, epoch_step=1, gamma=1, end_epoch=100, cfg=1)
        
        if epoch % 5 == 0:
            pass # TODO

    #     if (epoch >= 50 and epoch % 10 == 0):
    #         eval_net(
    #             val_dataset,
    #             val_loader,
    #             net,
    #             detector,
    #             cfg,
    #             ValTransform,
    #             top_k,
    #             thresh=thresh,
    #             batch_size=batch_size)
    # save_checkpoint(net, end_epoch, size, optimizer)

    


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        img = cv2.imread("/Users/zhubin/Documents/work_git/RetinaFace-pytorch/images/tensorrt_python_support.png")
        img = cv2.resize(img, (640, 640)).astype(np.float32)
        img = img[:, :, (2, 1, 0)] # bgr 2 rgb
        img = img.transpose(2, 0, 1) # (H,W,C) => (C,H,W)
        img = torch.from_numpy(img).unsqueeze(0)#.cuda()

        backbone = resnet50()
        model = RetinaFace(backbone)

        # fea = resnet50()#.eval()#.cuda()
        # model = FPN(fea)
        # print(model)
        fea_dict = model(img)
    Training = True
    if Training:
        main()
