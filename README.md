# MixFaceNets



## This is the official repository of the paper: PocketNet: Extreme Lightweight Face Recognition Network using Neural Architecture Search and  Multi-Step Knowledge Distillation
[arxiv](https://arxiv.org/abs/2108.10710)

!(evaluation)[logs/tradeoff.png]


### Model Training 
1. Train PocketNet with ArcFace loss
   + ./train.sh

2. Train PocketNet with template knowledge distillation
    + ./train_kd.sh
3. Train PocketNet with multi-step template knowledge distillation
    + ./train_kd.sh

| Model  | Pretrained model
| ------------- |  ------------- |
| PocketNetS-128 | [Pretrained-model](https://www.dropbox.com/sh/38mhqa19xx28438/AABw64kuY4ExrE4NAQLLiJJwa?dl=0)  |
| PocketNetS-256 |  [Pretrained-model](https://www.dropbox.com/sh/n2blqt17bg5eh1m/AAAxhWFZ2mC2hveuHzSMy0mma?dl=0) |
| PocketNetM-128 | [Pretrained-model](https://www.dropbox.com/sh/a8qgqkyryli0nl2/AABPlP5fmiZzlN8IV64BBGica?dl=0)  |
| PocketNetM-256 | [Pretrained-model](https://www.dropbox.com/sh/4dz14jgynrmsdgb/AAAsfYtKBXg1tPuK7RwzDbGva?dl=0)  |


### To-do 
- [x] Add pretrained model
- [ ] Add NAS code
- [ ] Add evaluation results
 
 
If you use any of the provided code in this repository, please cite the following paper:
```
@misc{boutros2021pocketnet,
      title={PocketNet: Extreme Lightweight Face Recognition Network using Neural Architecture Search and Multi-Step Knowledge Distillation}, 
      author={Fadi Boutros and Patrick Siebke and Marcel Klemt and Naser Damer and Florian Kirchbuchner and Arjan Kuijper},
      year={2021},
      eprint={2108.10710},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```