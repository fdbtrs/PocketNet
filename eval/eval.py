import logging
import os

#import cv2
import sys
import torch

#import backbones.mixnetm as mx
from backbones.augment_cnn import AugmentCNN 
import backbones.genotypes as gt

from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging

sys.path.append('/root/xy/work_dir/xyface/')
from util.config import config as cfg



if __name__ == "__main__":
    gpu_id = 0
    log_root = logging.getLogger()
    init_logging(log_root, 0, cfg.output,logfile="test.log")
    callback_verification = CallBackVerification(1, 0, cfg.val_targets, cfg.rec)
    output_folder='/home/psiebke/Thesis/dartFaceNetKD/outputKD/ForPaperDartFaceNet128Arc-KD'
    weights=os.listdir(output_folder)

    genotype = gt.from_str(cfg.genotypes["softmax_casia"])

    for w in weights:
        if "backbone" in w:

            backbone = AugmentCNN(C=16, n_layers=18, genotype=genotype, stem_multiplier=4, emb=cfg.embedding_size).to(f"cuda:{gpu_id}")

            backbone.load_state_dict(torch.load(os.path.join(output_folder,w)))
            model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
            callback_verification(int(w.split("backbone")[0]),model)

