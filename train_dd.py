import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time
import os
import os.path as osp

# from models.model_builder import RetinaFace
from models.retina_face import RetinaFace
# from models.retina_face import *
from models.resnet import *
from data.face_loader import WiderFaceDetection
from layers.modules import MultiTaskLoss
# from torch.optim import lr_scheduler

# from layers.modules import 
from layers.functions import Anchor_Box

from utils.utils import get_cur_time, adjust_learning_rate, detection_collate
import torch.optim as optim
import time
import argparse

from easydict import EasyDict as edict
__C = edict()
cfg = __C

# cfg.Landmark = True
cfg.FACE_LANDMARK = True
# cfg.MIN_FACE = 0
# cfg.USE_FLIPED = True



parser = argparse.ArgumentParser(description='RetinaFace')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--use_tensorboard', default=True, help='Log progress to TensorBoard')
parser.add_argument('-max','--max_epoch', default=150, type=int, help='max epoch for retraining')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--root', default="/workspace/mnt/group/algorithm/zhubin/cache_file/RetinaFace/data/retinaface", help='Dataset root directory path')
parser.add_argument('--dataset_root', default="/workspace/mnt/group/algorithm/zhubin/cache_file/RetinaFace/data/retinaface/train", help='Dataset root directory path')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--frequent', default=30, type=int, help='frequency of logging')
parser.add_argument('--pretrained', default="weights/pretrained/resnet50_official_pretrain.pth", help='Pretrained model path')
parser.add_argument('--arch', default="resnet50", type=str, help='Pretrained model path')

parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
parser.add_argument('--log_dir', default='log/', help='TensorBoard log directory ')

args = parser.parse_args()

#scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,60], gamma=0.1)




def eval_net(val_dataset, val_loader, net, detector, cfg, transform, max_per_image=300, thresh=0.01, batch_size=1):
    net.eval()



def train_net(train_loader, net, criterion, optimizer, epoch, anchors, train_writer=None):
    net.train()
    nums_id = 0
    # start_epoch = 0
    # for epoch in range(start_epoch+1, end_epoch+1):
        
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
    
    for i, (images, targets) in enumerate(train_loader):
        DEBUG_I = False
        if DEBUG_I:
            for ii in range(images.size(0)):
                image = images[ii]
                image = image.numpy()
                image = image.transpose(1, 2, 0)
                image = image[:, :, (2, 1, 0)]
                print(image.shape)
                cv2.imwrite("images/" + str(nums_id) + ".jpg", image)
                img = cv2.imread("images/" + str(nums_id) + ".jpg")

                bbxes = targets[0][ii]
                lmks = targets[1][ii]

                nums_id += 1
                print(bbxes.shape[0])
                for jj in range(bbxes.shape[0]):
                    sf, st = (int(bbxes[jj][0]), int(bbxes[jj][1])), (int(bbxes[jj][2]), int(bbxes[jj][3]))
                    print(sf, st)
                    print(lmks[jj])
                    cv2.rectangle(img, sf, st, (0, 0, 255), thickness=2)
                    cv2.circle(img,(lmks[jj][0, 0],lmks[jj][0, 1]),radius=1,color=(0,0,255),thickness=2)
                    cv2.circle(img,(lmks[jj][1, 0],lmks[jj][1, 1]),radius=1,color=(0,255,0),thickness=2)
                    cv2.circle(img,(lmks[jj][2, 0],lmks[jj][2, 1]),radius=1,color=(255,0,0),thickness=2)
                    cv2.circle(img,(lmks[jj][3, 0],lmks[jj][3, 1]),radius=1,color=(0,255,255),thickness=2)
                    cv2.circle(img,(lmks[jj][4, 0],lmks[jj][4, 1]),radius=1,color=(255,255,0),thickness=2)

                cv2.imwrite("images/" + str(nums_id) + "_0.jpg", img)

        if cfg.FACE_LANDMARK:
            boxes, landmarks = targets
        else:
            boxes = targets

        images = images.cuda()

        with torch.no_grad():
            boxes = [box.cuda() for box in boxes]
            if cfg.FACE_LANDMARK:
                landmarks = [lm.cuda() for lm in landmarks]
        output = net(images)
        optimizer.zero_grad()
        

        loss_conf, loss_loc, loss_landmark = criterion(output, anchors, [boxes, landmarks])
        # loss_conf, loss_loc = criterion(output, anchors, [boxes, landmarks])
        # loss_conf = criterion(output, anchors, [boxes, landmarks])

        loss = loss_conf[0] + loss_conf[1] + loss_conf[2] 
        loss += loss_loc[0] + loss_loc[1] + loss_loc[2] 
        loss += loss_landmark[0] + loss_landmark[1] + loss_landmark[2] 
        
        total_loss += loss.detach()
        loss_32_conf += loss_conf[0]
        loss_16_conf += loss_conf[1]
        loss_8_conf += loss_conf[2]

        loss_32_loc += loss_loc[0]
        loss_16_loc += loss_loc[1]
        loss_8_loc += loss_loc[2]

        loss_32_landmark += loss_landmark[0]
        loss_16_landmark += loss_landmark[1]
        loss_8_landmark += loss_landmark[2]

        if (i+1) % args.frequent == 0:
            print("Epoch[{}]  Batch [{}-{}]  total loss: {:4.6f}\t"\
            "32_conf: {:4.6f}\t16_conf: {:4.6f}\t8_conf: {:4.6f}\t" \
            "32_loc: {:4.6f}\t16_loc: {:4.6f}\t8_loc: {:4.6f}\t" \
            "32_landmark: {:4.6f}\t16_landmark: {:4.6f}\t8_landmark: {:4.6f}".format(epoch, 0, i+1, total_loss.item()/(i+1), \
            loss_32_conf.item()/(i+1), loss_16_conf.item()/(i+1), loss_8_conf.item()/(i+1), \
            loss_32_loc.item()/(i+1), loss_16_loc.item()/(i+1), loss_8_loc.item()/(i+1), \
            loss_32_landmark.item()/(i+1), loss_16_landmark.item()/(i+1), loss_8_landmark.item()/(i+1) ))

        # if (i+1) % args.frequent == 0:
        #     print("Epoch[{}]  Batch [{}-{}]  total loss: {:4.6f}\t"\
        #     "32_conf: {:4.6f}\t16_conf: {:4.6f}\t8_conf: {:4.6f}\t" \
        #     "32_loc: {:4.6f}\t16_loc: {:4.6f}\t8_loc: {:4.6f}".format(epoch, 0, i+1, total_loss.item()/(i+1), \
        #     loss_32_conf.item()/(i+1), loss_16_conf.item()/(i+1), loss_8_conf.item()/(i+1), \
        #     loss_32_loc.item()/(i+1), loss_16_loc.item()/(i+1), loss_8_loc.item()/(i+1) ))

        # if (i+1) % args.frequent == 0:
        #     print("Epoch[{}]  Batch [{}-{}]  total loss: {:4.6f}\t"\
        #     "32_conf: {:4.6f}\t16_conf: {:4.6f}\t8_conf: {:4.6f}".format(epoch, 0, i+1, total_loss.item()/(i+1), \
        #     loss_32_conf.item()/(i+1), loss_16_conf.item()/(i+1), loss_8_conf.item()/(i+1) ))
        
        loss.backward()
        optimizer.step()
    
    t_end = time.time()   
    print("Epoch[{}]  total loss: {:.6f}".format(epoch, total_loss.item()/(i+1)))  # :10.6f
    print("Epoch[{}]  loss_32_conf: {:.6f}".format(epoch, loss_32_conf.item()/(i+1)))
    print("Epoch[{}]  loss_16_conf: {:.6f}".format(epoch, loss_16_conf.item()/(i+1)))
    print("Epoch[{}]  loss_8_conf: {:.6f}".format(epoch, loss_8_conf.item()/(i+1)))
    print("Epoch[{}]  loss_32_loc: {:.6f}".format(epoch, loss_32_loc.item()/(i+1)))
    print("Epoch[{}]  loss_16_loc: {:.6f}".format(epoch, loss_16_loc.item()/(i+1)))
    print("Epoch[{}]  loss_8_loc: {:.6f}".format(epoch, loss_8_loc.item()/(i+1)))
    print("Epoch[{}]  loss_32_landmark: {:.6f}".format(epoch, loss_32_landmark.item()/(i+1)))
    print("Epoch[{}]  loss_16_landmark: {:.6f}".format(epoch, loss_16_landmark.item()/(i+1)))
    print("Epoch[{}]  loss_8_landmark: {:.6f}".format(epoch, loss_8_landmark.item()/(i+1)))
    print("Epoch[{}]  time comsuming: {}".format(epoch, (t_end - t_begin)))     
    if train_writer is not None:
        train_writer.add_scalar('total_loss', total_loss.item()/(i+1), epoch)
        train_writer.add_scalar('conf_s32_loss', loss_32_conf.item()/(i+1), epoch)
        train_writer.add_scalar('conf_s16_loss', loss_16_conf.item()/(i+1), epoch)
        train_writer.add_scalar('conf_s8_loss', loss_8_conf.item()/(i+1), epoch)
        train_writer.add_scalar('loc_s32_loss', loss_32_loc.item()/(i+1), epoch)
        train_writer.add_scalar('loc_s16_loss', loss_16_loc.item()/(i+1), epoch)
        train_writer.add_scalar('loc_s8_loss', loss_8_loc.item()/(i+1), epoch)
        train_writer.add_scalar('landmark_s32_loss', loss_32_landmark.item()/(i+1), epoch)
        train_writer.add_scalar('landmark_s16_loss', loss_16_landmark.item()/(i+1), epoch)
        train_writer.add_scalar('landmark_s8_loss', loss_8_landmark.item()/(i+1), epoch)

    # print("=========epoch {}=========".format(epoch+1))
    # print("time comsuming: {}".format((t2-t1)))     
    # print("total loss: {:.6f}".format(loss.item()/(epoch+1)))  # :10.6f
    # print("loss_32_conf: {:.6f}".format(loss_32_conf.item()/(epoch+1)))
    # print("loss_16_conf: {:.6f}".format(loss_16_conf.item()/(epoch+1)))
    # print("loss_8_conf: {:.6f}".format(loss_8_conf.item()/(epoch+1)))
    # print("loss_32_loc: {:.6f}".format(loss_32_loc.item()/(epoch+1)))
    # print("loss_16_loc: {:.6f}".format(loss_16_loc.item()/(epoch+1)))
    # print("loss_8_loc: {:.6f}".format(loss_8_loc.item()/(epoch+1)))
    # print("loss_32_landmark: {:.6f}".format(loss_32_landmark.item()/(epoch+1)))
    # print("loss_16_landmark: {:.6f}".format(loss_16_landmark.item()/(epoch+1)))
    # print("loss_8_landmark: {:.6f}".format(loss_8_landmark.item()/(epoch+1)))




    
def main():
    if args.arch == "resnet50":
        backbone = resnet50()
    elif args.arch == "resnet18":
        backbone = resnet18()
    else:
        raise NotImplementedError
    

    net = RetinaFace(backbone, pretrained_model_path=args.pretrained)
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

    if args.use_tensorboard:
        from tensorboardX import SummaryWriter
        if not osp.exists(args.log_dir):
            os.mkdir(args.log_dir)
        if args.log_dir:
            if not osp.exists(args.log_dir):
                os.mkdir(args.log_dir)
        train_writer = SummaryWriter(log_dir="{}".format(args.log_dir), comment=args.arch)
        
        # dummy_input = torch.rand(1, 3, 640, 640).cuda()
        # train_writer.add_graph(backbone, (dummy_input, ))

    train_dataset = WiderFaceDetection(root_path=args.root, data_path=args.dataset_root, phase="train", 
                                       dataset_name="WiderFace", transform=None)
    train_loader = data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate)

    print("sucess train_loader")

    anchors = Anchor_Box()
    with torch.no_grad():
        anchors = anchors.forward()
        anchors = anchors.cuda()
    print("anchors ready")
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    start_epoch = 0
    end_epoch = args.max_epoch
    criterion = MultiTaskLoss()
    for epoch in range(start_epoch + 1, end_epoch + 1):
        lr = adjust_learning_rate(optimizer=optimizer, 
                                  epoch=epoch, 
#                                   step_epoch=[55, 68, 80], 
                                  step_epoch=[55, 80, 100], 
                                  gamma=0.1, 
                                  base_lr=args.lr,  # 0.001
                                  warm_up_end_lr=0.01, 
#                                   warmup_epoch=5
                                 )
        print("Epoch[{}]  lr: {}".format(epoch, lr))
        if args.use_tensorboard:
            train_writer.add_scalar('learning_rate', lr, epoch)
            # train
            train_net(train_loader, net, criterion, optimizer, epoch, anchors, train_writer=train_writer)
        else:
            train_net(train_loader, net, criterion, optimizer, epoch, anchors)

        if epoch % 5 == 0:
            pass # TODO

    
        if (epoch == end_epoch) or (epoch % 5 == 0):
            torch.save(net.state_dict(), "weights/retinaface_epoch{}_{}.pth".format(epoch, get_cur_time()))
            # torch.save(net.state_dict(), "weights/retinaface_epoch{}_{}.pth".format(epoch, get_cur_time()))
    
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

    if args.use_tensorboard:
        train_writer.close()
    


if __name__ == "__main__":
    Training = True
    if Training:
        main()



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

    
