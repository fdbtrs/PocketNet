import torch
import torchvision
from facenet_pytorch import MTCNN
import cv2
from torch.utils.mobile_optimizer import optimize_for_mobile
import numpy as np
import backbones.genotypes as gt

from backbones.augment_cnn import AugmentCNN
from backbones.iresnet import iresnet100
from config.config_example import config as cfg

genotype = gt.from_str(cfg.genotypes["softmax_casia"])
model = AugmentCNN(C=cfg.channel, n_layers=cfg.n_layers, genotype=genotype, stem_multiplier=4,
                              emb=cfg.embedding_size).to('cpu')

model.eval()
example =  cv2.imread('test.jpg') #torch.rand(1, 3, 1024, 1024)
#example=np.transpose()
#traced_script_module = torch.jit.trace(model, torch.tensor(example))
#traced_script_module_optimized = optimize_for_mobile(traced_script_module)
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.ptl')

#torch.save(model,"model.pt")
