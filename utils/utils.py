import torch
import datetime

from easydict import EasyDict as edict
__C = edict()
cfg = __C

# cfg.Landmark = True
cfg.FACE_LANDMARK = True

def get_cur_time():
    '''
    Return: 
        Current time(str)
    '''
    now_time = datetime.datetime.now()
    year = str(now_time.year)
    month = str(now_time.month) if now_time.month>10  else ("0" + str(now_time.month))
    day = str(now_time.day) if now_time.day>10  else ("0" + str(now_time.day))
    hour = str(now_time.hour) if now_time.hour>10 else ("0" + str(now_time.hour))
    minute = str(now_time.minute) if now_time.minute>10 else ("0" + str(now_time.minute))
    cur_time = year + month + day + hour + minute
    return cur_time



def adjust_learning_rate(optimizer, epoch, step_epoch=[55, 68, 80], gamma=0.1, base_lr=0.001, warm_up_end_lr=0.01, warmup_epoch=None):
    assert epoch > 0
    if warmup_epoch is not None:
        if epoch <= warmup_epoch:
            assert (isinstance(warmup_epoch, int) and warmup_epoch>0)
            factor = pow((warm_up_end_lr / base_lr), 1.0/(warmup_epoch-1))
            # print("factor: ", factor)
            lr = base_lr * pow(factor, epoch-1)
        else:
            # factor = 0.0
            if epoch < step_epoch[0]:
                factor = 1.0
            elif epoch >= step_epoch[-1]:
                factor = gamma ** len(step_epoch)
            else:
                for k, v in enumerate(step_epoch):
                    if epoch>=step_epoch[k] and epoch <step_epoch[k+1]:
                        factor = gamma ** (k+1)
                        break
            lr = warm_up_end_lr * factor
    else:
        if epoch < step_epoch[0]:
            factor = 1.0
        elif epoch >= step_epoch[-1]:
                factor = gamma ** len(step_epoch)
        else:
            for k, v in enumerate(step_epoch):
                    if epoch>=step_epoch[k] and epoch <step_epoch[k+1]:
                        factor = gamma ** (k+1)
                        break
        lr = base_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # print(lr)
    return lr


def detection_collate(batch):
    '''
    Arguments:
        batch: batchsize 
    
    Return:
        Tuple including:
            1. batch image tenor: (N, C, H, W)
            2. batch bounding box ground truth: list type 
            3. batch landmark ground truth: list type
    '''
    images = []
    boxes = []
    if cfg.FACE_LANDMARK:
        landmarks = []
    for _, sample in enumerate(batch):
        # assert torch.is_tensor(sample[0])
        images.append(sample[0])
        boxes.append(torch.FloatTensor(sample[1]))
        if cfg.FACE_LANDMARK:
            landmarks.append(torch.FloatTensor(sample[2]))
        # boxes.append(torch.from_numpy(sample[1]).float())
        # landmarks.append(torch.from_numpy(sample[2]).float())
    if cfg.FACE_LANDMARK:
        return (torch.stack(images, 0), (boxes, landmarks))
    else:
        return (torch.stack(images, 0), boxes)


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


#######################invliad code ######################

def adjust_learning_rate_(optimizer, epoch, step_epoch, gamma, epoch_size, iteration):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    ## warmup
    if epoch <= cfg.TRAIN.WARMUP_EPOCH:
        if cfg.TRAIN.WARMUP:
            iteration += (epoch_size * (epoch - 1))
            lr = 1e-6 + (cfg.SOLVER.BASE_LR - 1e-6) * iteration / (
                epoch_size * cfg.TRAIN.WARMUP_EPOCH)
        else:
            lr = cfg.SOLVER.BASE_LR
    else:
        div = 0
        if epoch > step_epoch[-1]:
            div = len(step_epoch) - 1
        else:
            for idx, v in enumerate(step_epoch):
                if epoch > step_epoch[idx] and epoch <= step_epoch[idx + 1]:
                    div = idx
                    break
        lr = cfg.SOLVER.BASE_LR * (gamma**div)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr