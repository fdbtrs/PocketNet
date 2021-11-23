""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB and counted """
    #n_params = sum(
    #    np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return n_params / 1024. / 1024., n_params


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print(pred)
    #print(target)
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # .view(-1)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        #print(correct_k)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def save_checkpoint_search(epoch, model, optimizerW, optimizerA, loss, ckpt_dir, is_best=False):
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_w_state_dict': optimizerW.state_dict(),
        'optimizer_a_state_dict': optimizerA.state_dict(),
        'loss': loss,
    }, filename)

    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)

def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)