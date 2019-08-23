import torch
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from Detection import Detect

from easydict import EasyDict as edict
__C = edict()
cfg = __C

cfg.FACE_LANDMARK = True
# cfg.SCALES = (640, 640)
# cfg.CLIP = False
# cfg.RPN_FEAT_STRIDE = [8, 16, 32]



parser = argparse.ArgumentParser(description='Retinaface Testing')
# parser.add_argument('-c', '--config', default='configs/m2det320_vgg.py', type=str)
# parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO version')
# parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
# parser.add_argument('--test', action='store_true', help='to submit a test file')
parser.add_argument('--score_thresh', default=0.01, type=float, help='Score threshold for classification')
parser.add_argument('--gpu', default=True, help='use gpu or not')
args = parser.parse_args()


# anchors = Prior_Box()

# with torch.no_grad():
#     anchors = anchors.forward()
#     if args.gpu:
#         anchors = anchors.cuda()
# print("anchors ready")


def image_forward(img, net, cuda, priors, detector, transform):
    w,h = img.shape[1],img.shape[0]
    scale = torch.Tensor([w,h,w,h])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    out = net(x)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0] * scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    return boxes, scores
    

def test_net(net, testset):

    net.eval()
    test_image_nums = len(testset)
    detector = Detect()  # TODO
    num_classes = 2 # TODO
    all_boxes = [[[] for _ in range(test_image_nums)] for _ in range(num_classes)]
    # all_landmarks = [[[] for _ in range(test_image_nums)] for _ in range(num_classes)]

    for idx in tqdm(range(test_image_nums)):
        image = testset.pull_image(idx)  #  TODO

        target_size = 1600
        max_size = 2150
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
            im_tensor = np.zeros((3, im.shape[0], im.shape[1]))
            for i in range(3):
                im_tensor[i, :, :] = (im[:, :, 2 - i]/PIXEL_SCALE - PIXEL_MEANS[2 - i]) / PIXEL_STDS[2 - i]
            image_new = image_new.astype(np.float32)
            im_tensor = torch.from_numpy(image)
            net_out = net(im_tensor)

            if cfg.FACE_LANDMARK:
                scores, boxes, landmarks = detector.forward(net_out)
            else:
                scores, boxes = detector.forward(net_out)

            boxes = (boxes / im_scale).cpu().numpy()
            if cfg.FACE_LANDMARK:
                landmarks = (landmarks / im_scale).cpu().numpy()

            # TODO split as a function
            for cls in range(1: num_classes):
                inds = np.where(scores[:, cls] > args.score_thresh)[0] 
                if len(inds) == 0:
                    all_boxes[j][idx] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_boxes = boxes[inds]
                c_scores = scores[inds, j]
                c_dets = np.hstack((c_boxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)

                keep = nms(c_dets, args.nms_overlap, force_cpu=True)  # TODO
                box_num = 50
                keep = keep[: box_num] # keep only the highest boxes
                c_dets = c_dets[keep, :]
                all_boxes[j][idx] = c_dets
                




                

           


            




    




    


def main():
    if args.gpu:
        net.cuda()
    
    # test_net(net, dataloader, )


if __name__ == "__main__":
    test_net(None, None)

