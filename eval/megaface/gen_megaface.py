from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import numpy as np
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import torch

from util.config import config as cfg

from backbones.augment_cnn import AugmentCNN 
import backbones.genotypes as gt 



def read_img(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = (img / 255 - 0.5) / 0.5
    return img


def get_feature(imgs, nets, ctx):
    count = len(imgs)
    data = np.zeros(shape=(count * 2, 3, imgs[0].shape[0],
                              imgs[0].shape[1]))
    for idx, img in enumerate(imgs):
        img = img[:, :, ::-1]  #to rgb
        img = np.transpose(img, (2, 0, 1))
        for flipid in [0, 1]:
            _img = np.copy(img)
            if flipid == 1:
                _img = _img[:, :, ::-1]
            _img = np.array(_img)
            data[count * flipid + idx] = _img

    with torch.no_grad():
        F = []
        for net in nets:
            batch = torch.tensor(data).float().to(ctx)
            batch_emb = net(batch).cpu()
            embedding = batch_emb[0:count, :] + batch_emb[count:, :]
            embedding = sklearn.preprocessing.normalize(embedding)
            #print('emb', embedding.shape)
            F.append(embedding)
    F = np.concatenate(F, axis=1)
    F = sklearn.preprocessing.normalize(F)
    #print('F', F.shape)
    return F


def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))


def get_and_write(buffer, nets, ctx):
    imgs = []
    for k in buffer:
        imgs.append(k[0])
    features = get_feature(imgs, nets, ctx)
    #print(np.linalg.norm(feature))
    assert features.shape[0] == len(buffer)
    for ik, k in enumerate(buffer):
        out_path = k[1]
        feature = features[ik].flatten()
        write_bin(out_path, feature)


def main(args):

    print(args)
    gpuid = args.gpu
    if gpuid == -1:
        ctx = torch.device("cpu")
    else:
        ctx = torch.device(f"cuda:{gpuid}")
    nets = []
    image_shape = [int(x) for x in args.image_size.split(',')]

    for model_path in args.model.split('|'):
        genotype = gt.from_str(cfg.genotypes["softmax_casia"])
        backbone = AugmentCNN(C=32, n_layers=9, genotype=genotype, stem_multiplier=4, emb=cfg.embedding_size).to("cuda:0")

        backbone.load_state_dict(torch.load(model_path))
        #model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
        model = backbone.to(ctx)
        model.eval()
        print(f"LOADED MODEL FROM {model_path}")
        nets.append(model)

    facescrub_lst = os.path.join(args.megaface_data, 'facescrub_lst')
    facescrub_root = os.path.join(args.megaface_data, 'facescrub_images')
    facescrub_out = os.path.join(args.output, 'facescrub')
    megaface_lst = os.path.join(args.megaface_data, 'megaface_lst')
    megaface_root = os.path.join(args.megaface_data, 'megaface_images')
    megaface_out = os.path.join(args.output, 'megaface')

    i = 0
    succ = 0
    buffer = []
    for line in open(facescrub_lst, 'r'):
        if i % 1000 == 0:
            print("writing fs", i, succ)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a, b = _path[-2], _path[-1]
        out_dir = os.path.join(facescrub_out, a)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        image_path = os.path.join(facescrub_root, image_path)
        img = read_img(image_path)
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, nets, ctx)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, nets, ctx)
        buffer = []
    print('fs stat', i, succ)

    i = 0
    succ = 0
    buffer = []
    for line in open(megaface_lst, 'r'):
        if i % 1000 == 0:
            print("writing mf", i, succ)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        out_dir = os.path.join(megaface_out, a1, a2)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            #continue
        #print(landmark)
        image_path = os.path.join(megaface_root, image_path)
        img = read_img(image_path)
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, nets, ctx)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, nets, ctx)
        buffer = []
    print('mf stat', i, succ)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=8)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--gpu', type=int, help='', default=-1)
    parser.add_argument('--algo', type=str, help='', default='insightface')
    parser.add_argument('--megaface-data', type=str, help='', default='./data')
    parser.add_argument('--facescrub-lst',
                        type=str,
                        help='',
                        default='./data/facescrub_lst')
    parser.add_argument('--megaface-lst',
                        type=str,
                        help='',
                        default='./data/megaface_lst')
    parser.add_argument('--facescrub-root',
                        type=str,
                        help='',
                        default='./data/facescrub_images')
    parser.add_argument('--megaface-root',
                        type=str,
                        help='',
                        default='./data/megaface_images')
    parser.add_argument('--output', type=str, help='', default='./feature_out')
    parser.add_argument('--model', type=str, help='', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
