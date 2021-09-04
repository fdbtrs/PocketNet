""" Config class for search/augment """
import argparse
import os
from models import genotypes as gt
from functools import partial

from easydict import EasyDict as edict

config = edict()
config.name = "searchPocketNet"
config.batch_size = 128
config.w_lr = 0.1
config.w_lr_min = 0.004
config.w_momentum = 0.9
config.w_weight_decay = 3e-4
config.w_grad_clip = 5.
config.print_freq = 50
config.epochs = 50
config.init_channels = 16
config.layers = 8
config.seed = 2
config.workers = 4
config.alpha_lr = 12e-4
config.alpha_weight_decay = 1e-3
config.input_channels = 3
config.stem_multiplier = 3
config.n_nodes = 4

config.dataset = "CASIA" # CASIA | CIFAR-10

config.path = os.path.join('searchs_output', config.name)
config.plot_path = os.path.join(config.path, 'plots')

if config.dataset == "CASIA":
    config.input_size = 112
    config.root = ""
    config.n_classes = 10572
elif config.dataset == "CIFAR-10":
    config.input_size = 32
    config.root = ""
    config.n_classes = 10000


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser

# Base class to store and print configurations
def print_params(prtf=print):
    """ prints configs """
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(config.items()):
        prtf("{}={}".format(attr.upper(), value))
    prtf("")

def as_markdown():
    """ Returns configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(config.items()):
        text += "|{}|{}|  \n".format(attr, value)
    
    return text