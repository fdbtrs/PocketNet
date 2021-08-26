""" Search Cell """
import os
import torch
import torch.nn as nn
import numpy as np
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
from util.config import SearchConfig
from util import dataset as dataset
from models.search_cnn import SearchCNNController
from searchs.architect import Architect
from util.visualize import plot

# get configurations
config = SearchConfig()

# set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # maybe this is redundant

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text("config", config.as_markdown(), 0)

# logger
logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

def main():
    logger.info("Logger is set - training start")

    if device == "cuda":
        # set default gpu device id
        torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

        # some optimization
        torch.backends.cudnn.benchmark = True # maybe not nescessary???
    
    # get dataset and meta info
    input_size, input_channels, n_classes, train_data = dataset.get_train_dataset(config.dataset, reduced=False)
    val_data = dataset.get_casia_without_crop(reduced=False)
    # assume that indices of train_data and val_data are the same

    # split into train val and get indices of splits
    train_idx, val_idx = dataset.get_train_val_split(train_data, 0.5)

    # setup model
    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers, net_crit, device_ids=config.gpus)
    model = model.to(device)

    

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)

    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999), weight_decay=config.alpha_weight_decay)

    # loader for train and val data
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=config.workers,
        pin_memory=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=config.workers,
        pin_memory=False,
        drop_last=False
    )


    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min
    )

    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        #lr = lr_scheduler.get_lr()[0]
        lr = lr_scheduler.get_last_lr()[0]

        model.print_alphas(logger)

        # training
        train(train_loader, val_loader, model, architect, w_optim, alpha_optim, lr, epoch)

        lr_scheduler.step()

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(val_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # always know which was the best cell (prevent overfitting???, kind of early stopping)
        # save
        if best_top1 <= top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        #utils.save_checkpoint(model, config.path, is_best)
        utils.save_checkpoint_search(epoch, model, w_optim, alpha_optim, top1, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))

def train(train_loader, val_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, val_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        #alpha_optim.step() # change position because of pytorch warning

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        alpha_optim.step()

        prec1 = utils.accuracy(logits, trn_y, topk=(1,))
        prec1 = prec1[0]

        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1) ({top1.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(val_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(val_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1 = utils.accuracy(logits, y, topk=(1,))
            prec1 = prec1[0]

            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)

            if step % config.print_freq == 0 or step == len(val_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(val_loader)-1, losses=losses,
                        top1=top1))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg

if __name__ == "__main__":
    main()