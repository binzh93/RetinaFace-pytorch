import torch
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from layers.functions import Detect


from models.resnet import *
from models.retina_face import *
from utils.nms_wrapper import nms, soft_nms

from configs.config import cfg

# from easydict import EasyDict as edict
# __C = edict()
# cfg = __C

# cfg.FACE_LANDMARK = False
# cfg.SCALES = (640, 640)
# cfg.CLIP = False
# cfg.RPN_FEAT_STRIDE = [8, 16, 32]



parser = argparse.ArgumentParser(description='Retinaface Testing')
# parser.add_argument('-c', '--config', default='configs/m2det320_vgg.py', type=str)
# parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO version')
# parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
# parser.add_argument('--test', action='store_true', help='to submit a test file')
# parser.add_argument('--model_path', default="weights/retinaface_epoch100_201908220041.pth", type=str, help='Score threshold for classification')
parser.add_argument('--model_path', default="weights/retinaface_epoch10_201909141420.pth", type=str, help='Score threshold for classification')
parser.add_argument('--score_thresh', default=0.351, type=float, help='Score threshold for classification')
parser.add_argument('--nms_overlap', default=0.4, type=float, help='NMS Score threshold for classification')

parser.add_argument('--gpu', default=False, help='use gpu or not')
args = parser.parse_args()

    

def test_net(net, testset):

    net.eval()
    test_image_nums = len(testset)
    detector = Detect()  # TODO
    num_classes = 2 # TODO
    all_boxes = [[[] for _ in range(test_image_nums)] for _ in range(num_classes)]
    all_landmarks = [[[] for _ in range(test_image_nums)] for _ in range(num_classes)]
    # all_landmarks = [[[] for _ in range(test_image_nums)] for _ in range(num_classes)]
    print(all_boxes)
    for idx in tqdm(range(test_image_nums)):
        with torch.no_grad():
            # image = testset.pull_image(idx)  #  TODO
            image = cv2.imread(testset[idx])   #  TODO

            target_size = 1600
            max_size = 2150
            # target_size = 2000
            # max_size = 3000
            # target_size = 640
            # max_size = 900
            # target_size = 640
            # max_size = 640
            im_shape = image.shape  # H, W, C
            
            im_size_min = min(im_shape[0: 2])
            im_size_max = max(im_shape[0: 2])
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            scales = [im_scale]

            for im_scale in scales:
                if im_scale != 1.0:
                    image_new = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                else:
                    image_new = image.copy()
                PIXEL_MEANS = np.array([0.406,0.456, 0.485])  # bgr mean
                PIXEL_STDS = np.array([0.225, 0.224, 0.229])
                PIXEL_SCALE = 255.0
                im_tensor = np.zeros((3, image_new.shape[0], image_new.shape[1]))
                for i in range(3):
                    im_tensor[i, :, :] = (image_new[:, :, 2 - i]/PIXEL_SCALE - PIXEL_MEANS[2 - i]) / PIXEL_STDS[2 - i]
                im_tensor = im_tensor.astype(np.float32)
                print("im_tensor: ", im_tensor.shape)
                im_tensor = torch.from_numpy(im_tensor)
                im_tensor = im_tensor.unsqueeze(0)
                if args.gpu:
                    im_tensor = im_tensor.cuda()
                net_out = net(im_tensor)

                if cfg.FACE_LANDMARK:
                    print("im_tensor222: ", im_tensor.shape[2: ])
                    scores, boxes, landmarks = detector.forward(net_out, im_tensor.shape[2: ])
                    # scores, boxes, landmarks = detector.forward(net_out)
                else:
                    scores, boxes = detector.forward(net_out, im_tensor.shape[2: ])
                scores = scores.cpu().numpy()
                boxes = boxes.cpu().numpy() / im_scale
                # boxes = boxes.cpu().numpy()
                #.cpu().numpy()
                if cfg.FACE_LANDMARK:
                    landmarks = landmarks.cpu().numpy() / im_scale #.cpu().numpy()
                    # landmarks = landmarks.cpu().numpy()
                print(scores.shape)
                print(boxes.shape)
                # TODO split as a function
                for cls in range(1, num_classes):
                    inds = np.where(scores[:, cls] > args.score_thresh)[0] 
                    if len(inds) == 0:
                        print("XXXXXXX")
                        all_boxes[cls][idx] = np.empty([0, 5], dtype=np.float32)
                        if cfg.FACE_LANDMARK:
                            all_landmarks[cls][idx] = np.empty([0, 10], dtype=np.float32)
                        continue
                    c_boxes = boxes[inds]
                    c_scores = scores[inds, cls]
                    c_dets = np.hstack((c_boxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                    if cfg.FACE_LANDMARK:
                        c_landmarks = landmarks[inds]
                    # print(c_dets)
                    keep = nms(c_dets, args.nms_overlap, force_cpu=True)  # TODO   soft_nms
                    box_num = 150 #50
                    keep = keep[: box_num] # keep only the highest boxes
                    c_dets = c_dets[keep, :]
                    all_boxes[cls][idx] = c_dets
                    if cfg.FACE_LANDMARK:
#                         c_landmarks = c_landmarks[keep, :]
                        all_landmarks[cls][idx] = c_landmarks
                
                bbx = all_boxes[1][0]
                lmks = all_landmarks[1][0]
            print(lmks)
#             print(bbx)
            DEBUG_I = True
            if DEBUG_I:
                # img = cv2.imread(roi['image_path'])      
                # bbxes = roi['boxes']
                # lmks = roi['landmarks']#.reshape(-1, 15)
                # print(roi['image_path'])
                # if roi['flipped']:
                #     img = img[:, ::-1]
                #     print("images/z_flipped_" + osp.basename(roi['image_path']))
                #     cv2.imwrite("images/z_flipped_" + osp.basename(roi['image_path']), img)
                #     img = cv2.imread("images/z_flipped_" + osp.basename(roi['image_path']))
                
            
                for jj in range(bbx.shape[0]):
                    sf, st = (int(bbx[jj][0]), int(bbx[jj][1])), (int(bbx[jj][2]), int(bbx[jj][3]))
                    print(sf, st)
                    # print(lmks[jj])
                    # print()
                    cv2.rectangle(image, sf, st, (0, 0, 255), thickness=2)
                    # print((lmks[jj][0, 0],lmks[jj][0, 1]))
                    # print((lmks[jj][1, 0],lmks[jj][1, 1]))
                    # print((lmks[jj][2, 0],lmks[jj][2, 1]))
                    # print((lmks[jj][3, 0],lmks[jj][3, 1]))
                    # print((lmks[jj][4, 0],lmks[jj][4, 1]))
#                     cv2.circle(image,(lmks[jj][0, 0],lmks[jj][0, 1]),radius=1,color=(0,0,255),thickness=2)
#                     cv2.circle(image,(lmks[jj][1, 0],lmks[jj][1, 1]),radius=1,color=(0,255,0),thickness=2)
#                     cv2.circle(image,(lmks[jj][2, 0],lmks[jj][2, 1]),radius=1,color=(255,0,0),thickness=2)
#                     cv2.circle(image,(lmks[jj][3, 0],lmks[jj][3, 1]),radius=1,color=(0,255,255),thickness=2)
#                     cv2.circle(image,(lmks[jj][4, 0],lmks[jj][4, 1]),radius=1,color=(255,255,0),thickness=2)

                    # cv2.circle(image,(lmks[jj][0],lmks[jj][1]),radius=1,color=(0,0,255),thickness=2)
                    # cv2.circle(image,(lmks[jj][2],lmks[jj][3]),radius=1,color=(0,255,0),thickness=2)
                    # cv2.circle(image,(lmks[jj][4],lmks[jj][5]),radius=1,color=(255,0,0),thickness=2)
                    # cv2.circle(image,(lmks[jj][6],lmks[jj][7]),radius=1,color=(0,255,255),thickness=2)
                    # cv2.circle(image,(lmks[jj][8],lmks[jj][9]),radius=1,color=(255,255,0),thickness=2)
                cv2.imwrite("images/img.jpg", image)
                # if roi['flipped']:
                #     cv2.imwrite("images/flipped_" + osp.basename(roi['image_path']), img)
                # else:
                #     cv2.imwrite("images/" + osp.basename(roi['image_path']), img)
                    


def main():
    
    backbone = resnet50()
    net = RetinaFace(backbone)

    # print(net)
    print("-----here----")
    
    net_weights = torch.load(args.model_path, map_location='cpu')
    
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in net_weights.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    


    # import pdb
    # pdb.set_trace()
    # for i in net_weights.keys():
    #     print(i)

    # net.load_state_dict(net_weights)
    # print(net_weights)
    
    # net.load_state_dict(net_weights)

    if args.gpu:
        net.cuda()
    
    # dataloader = ['/home/dc2-user/zhubin/wider_face/val/images/12--Group/12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_868.jpg']
#     dataloader = ['/workspace/mnt/group/algorithm/zhubin/cache_file/RetinaFace/data/retinaface/train/images/15--Stock_Market/15_Stock_Market_Stock_Market_15_479.jpg']
#     dataloader = ['/workspace/mnt/group/algorithm/zhubin/cache_file/RetinaFace/data/retinaface/train/images/0--Parade/0_Parade_marchingband_1_1031.jpg']
    # dataloader = ['/data/zhubin/wider_face/val/images/12--Group/12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_868.jpg']
    
    # dataloader = ['images/49_Greeting_peoplegreeting_49_992.jpg']
    # dataloader = ['images/49_Greeting_peoplegreeting_49_877.jpg']
    # dataloader = ['images/59_peopledrivingcar_peopledrivingcar_59_925.jpg']
    # dataloader = ['images/ju1.jpeg']
    dataloader = ['images/timg.jpeg']
    # dataloader = ['images/聚餐_68.jpg']
    # dataloader = ['images/聚餐_24.jpg']
    # dataloader = ['images/聚餐_47.jpeg']
    # dataloader = ['images/61_Street_Battle_streetfight_61_925.jpg']
    # dataloader = ['images/61_Street_Battle_streetfight_61_5.jpg']
   
    

    
    test_net(net, dataloader)


if __name__ == "__main__":
    main()

