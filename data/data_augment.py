import torch
import cv2
import numpy as np
# import random
import os.path as osp

import copy


from easydict import EasyDict as edict
__C = edict()
cfg = __C
# cfg.SCALES = [(640, 640)]
cfg.SCALES = (640, 640)
cfg.FACE_LANDMARK = True
cfg.MIN_FACE = 0
cfg.COLOR_JITTERING = 0.125
cfg.USE_BLUR = False

def transform_ms(im, pixel_means, pixel_stds, pixel_scale):
    """
    transform into mxnet tensor,
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[i, :, :] = (im[:, :, 2 - i]/pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]
    return im_tensor

def transform_ms_old(im, pixel_means, pixel_stds, pixel_scale):
    """
    transform into mxnet tensor,
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = (im[:, :, 2 - i]/pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]
    return im_tensor


def crop_img(roi):
    if 'image_data' in roi:
        image = roi['image_data']
    else:
        assert osp.exists(roi['image_path'])
        image = cv2.imread(roi['image_path'])
        if roi['flipped']:
            image = image[:, ::-1]
#             roi['flipped'] = False
            
    DEBUG = False
    if DEBUG:
        image__ = image.copy()
        print("img path: ", roi['image_path'])
        print(image.shape)      
        for i in range(roi['boxes'].shape[0]):
            print((int(roi['boxes'][i][0]), int(roi['boxes'][i][1])), (int(roi['boxes'][i][2]), int(roi['boxes'][i][3])))
            cv2.rectangle(image__, (int(roi['boxes'][i][0]), int(roi['boxes'][i][1])), (int(roi['boxes'][i][2]), int(roi['boxes'][i][3])), (0,255,0), 3)
        cv2.imwrite("images/" + osp.basename(roi['image_path']), image__)

    # TODO
    INPUT_SIZE = cfg.SCALES[0]
    PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
    _scale = np.random.choice(PRE_SCALES) # TODO best ==> np.random(0.3, 1.0) ????
    size = int(np.min(image.shape[0:2])*_scale)
    im_scale = float(INPUT_SIZE)/size
    # origin_shape = image.shape
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    assert image.shape[0]>=INPUT_SIZE and image.shape[1]>=INPUT_SIZE
    #print('image size', origin_shape, _scale, SIZE, size, im_scale)

    boxes = roi['boxes'].copy()
    boxes = boxes * im_scale
    if cfg.FACE_LANDMARK:
        landmarks = roi['landmarks'].copy()
        landmarks[:, :, 0: 2] = landmarks[:, :, 0: 2] * im_scale
    
    LIMITED_TIMES = 30
    # LIMITED_TIMES = 1000
    FLAG = 0
    retry = 0
#     while retry<LIMITED_TIMES:
    while True:
        # cv2 shape ==> (H, W, C)
        x_tl, y_tl = (np.random.randint(0, image.shape[1]-INPUT_SIZE+1), np.random.randint(0, image.shape[0]-INPUT_SIZE+1))
        
        image_new = image[y_tl: (y_tl+INPUT_SIZE), x_tl: (x_tl+INPUT_SIZE), :]
        image_h, image_w, image_c = image_new.shape
        boxes_new = boxes.copy()
        boxes_new[:, 0] -= x_tl
        boxes_new[:, 1] -= y_tl
        boxes_new[:, 2] -= x_tl
        boxes_new[:, 3] -= y_tl

        if cfg.FACE_LANDMARK:
            landmarks_new = landmarks.copy()
            landmarks_new[:, :, 0] -= x_tl
            landmarks_new[:, :, 1] -= y_tl
            valid_landmarks = []
        
        valid_inds = []
        valid_boxes = []

        for idx in range(boxes_new.shape[0]):
            # print("crop-----------")
            box = boxes_new[idx]

            centerx = (box[0]+box[2])/2
            centery = (box[1]+box[3])/2
            box_size = max(box[2]-box[0], box[3]-box[1])

            # filter box
            if centerx<0 or centery<0 or centerx>=image_w or centery>=image_h:
                continue
            if box_size<=cfg.MIN_FACE:
                continue

            valid_inds.append(idx)
            valid_boxes.append(box)
            if cfg.FACE_LANDMARK:
                valid_landmarks.append(landmarks_new[idx])
#         if len(valid_inds)>0 or retry==(LIMITED_TIMES-1):
        if len(valid_inds)>0:
            FLAG = 1
            image = image_new
            boxes = np.array(valid_boxes, dtype=np.float32)
            gt_classes = roi['gt_classes'][valid_inds]

            roi_crop = {
                'image_path': roi['image_path'], 
                'height': INPUT_SIZE,
                'width': INPUT_SIZE,
                'boxes': boxes,   # x1, y1, x2, y2
                'gt_classes': gt_classes,
                'blur': roi['blur'].copy(),
                'flipped': roi['flipped']
            }
            # if "image_data" in roi:
            #     roi_crop['image_data'] = roi['image_data'].copy()
            # MUST SAVE IMAGE_DATA or SAVE RESIZE SCALE
            roi_crop['image_data'] = image  
            roi_crop['flipped'] = False
            if cfg.FACE_LANDMARK:
                # landmarks = valid_landmarks
                valid_landmarks = np.array(valid_landmarks, dtype=np.float32)
                roi_crop['landmarks'] = valid_landmarks
            break
        retry += 1
    
    
    # if cfg.COLOR_JITTERING>0.0:
    #     pass
        # im = im.astype(np.float32)
        # im = color_aug(im, config.COLOR_JITTERING)
    # im_tensor = transform(im, config.PIXEL_MEANS, config.PIXEL_STDS, config.PIXEL_SCALE)
    # processed_ims.append(im_tensor)
    # im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
    # new_rec['im_info'] = np.array(im_info, dtype=np.float32)
    # processed_roidb.append(new_rec)
    if FLAG==0: 
        print("Fail crop")
    else:
        DEBUG = False
        # DEBUG = True
        if DEBUG:
            image = roi_crop['image_data'].copy()
#             print(image.shape)
            
            for i in range(roi_crop['boxes'].shape[0]):
#                 print((int(roi_crop['boxes'][i][0]), int(roi_crop['boxes'][1])), int(roi_crop['boxes'][2]), int(roi_crop['boxes'][3]))
                cv2.rectangle(image, (int(roi_crop['boxes'][i][0]), int(roi_crop['boxes'][i][1])), (int(roi_crop['boxes'][i][2]), int(roi_crop['boxes'][i][3])), (0,255,0), 3)
            if roi['flipped']:
                cv2.imwrite("images/flipped_" + osp.basename(roi_crop['image_path']), image)
            else:
                cv2.imwrite("images/" + osp.basename(roi_crop['image_path']), image)
#             print(roi_crop['boxes'])
#             cv2.rectangle(im,(int(sx1),int(sy1)),(int(sx2),int(sy2)),(0,255,0),3)
#             函数参数： 图片， 左上角， 右下角， 颜色， 线条粗细， 线条类型，点类型
    #if retry >LIMITED_TIMES:
            #print("{}, crop times: {}".format(roi['image_path'], retry))
    return roi_crop
            

# class Resize(object):
#     def __init__(self, ):
#         pass
        
#     def __call__(self, image, boxes=None, landmark=None, labels=None):
#         return image

# class 

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, landmark=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, landmark, labels


class RandomBrightness(object):
    def __init__(self, delta=0.125):
        assert delta >= 0.0
        assert delta <= 1.0
        self.delta = delta

    def __call__(self, image, boxes=None, landmarks=None, labels=None):
        alpha = 1.0 + np.random.uniform(-self.delta, self.delta)
        image *= alpha
        return image, boxes, landmarks, labels
    

class RandomContrast(object):
    def __init__(self, delta=0.125):
        assert delta >= 0.0
        assert delta <= 1.0
        self.delta = delta

    def __call__(self, image, boxes=None, landmarks=None, labels=None):
        alpha = 1.0 + np.random.uniform(-self.delta, self.delta)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image *= alpha
        image += gray
        return image, boxes, landmarks, labels
    

class RandomSaturation(object):
    def __init__(self, delta=0.125):
        assert delta >= 0.0
        assert delta <= 1.0
        self.delta = delta

    def __call__(self, image, boxes=None, landmarks=None, labels=None):
        alpha = 1.0 + np.random.uniform(-self.delta, self.delta)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = image * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        image *= alpha
        image += gray
        return image, boxes, landmarks, labels


def color_aug(img, x):
  if config.COLOR_MODE>1:
    augs = [brightness_aug, contrast_aug, saturation_aug]
    random.shuffle(augs)
  else:
    augs = [brightness_aug]
  for aug in augs:
    #print(img.shape)
    img = aug(img, x)
    #print(img.shape)
  return img
