# PocketNet



## This is the official repository of the paper:
#### PocketNet: Extreme Lightweight Face Recognition Network using Neural Architecture Search and  Multi-Step Knowledge Distillation
Paper on arxiv: [arxiv](https://arxiv.org/abs/2108.10710)

![evaluation](https://raw.githubusercontent.com/fdbtrs/PocketNet/main/logs/tradeoff.png)


### Face recognition  model training 
Download [MS1MV2](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view?usp=sharing) dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) on strictly follow the licence distribution
Eaxtract the dataset and place it in the data folder
1. Train PocketNet with ArcFace loss
   + ./train.sh
2. Train PocketNet with template knowledge distillation
    + ./train_kd.sh
3. Train PocketNet with multi-step template knowledge distillation
    + ./train_kd.sh

| Model  | Parameters (M)| configuration | log| pretrained model| 
| ------------- | ------------- |  ------------- |------------- |------------- |
| PocketNetS-128 |0.92 |[Config](https://github.com/fdbtrs/PocketNet/blob/main/config/config_PocketNetS128.py)| [log](https://www.dropbox.com/s/hha0qp63y8w46ng/training.log?dl=0)|[Pretrained-model](https://www.dropbox.com/sh/38mhqa19xx28438/AABw64kuY4ExrE4NAQLLiJJwa?dl=0)  |
| PocketNetS-256 |0.99 |[Config](https://github.com/fdbtrs/PocketNet/blob/main/config/config_PocketNetS256.py)| [log](https://www.dropbox.com/s/tenmtzjrghaos75/training.log?dl=0)|[Pretrained-model](https://www.dropbox.com/sh/n2blqt17bg5eh1m/AAAxhWFZ2mC2hveuHzSMy0mma?dl=0) |
| PocketNetM-128 |1.68 |[Config](https://github.com/fdbtrs/PocketNet/blob/main/config/config_PocketNetM128.py) | [log](https://www.dropbox.com/s/o0vnxns6hmmj1rg/training.log?dl=0)|[Pretrained-model](https://www.dropbox.com/sh/a8qgqkyryli0nl2/AABPlP5fmiZzlN8IV64BBGica?dl=0)  |
| PocketNetM-256 |1.75 |[Config](https://github.com/fdbtrs/PocketNet/blob/main/config/config_PocketNetM256.py)| [log](https://www.dropbox.com/s/lqs47v4rc5g7425/training.log?dl=0) |[Pretrained-model](https://www.dropbox.com/sh/4dz14jgynrmsdgb/AAAsfYtKBXg1tPuK7RwzDbGva?dl=0)  |

### Differentiable architecture search training
### To-do 
- [x] Add pretrained model
- [x] Training configuration
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