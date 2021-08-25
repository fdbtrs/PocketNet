# MixFaceNets



## This is the official repository of the paper: PocketNet: Extreme Lightweight Face Recognition Network using Neural Architecture Search and  Multi-Step Knowledge Distillation
[arxiv](https://arxiv.org/abs/2108.10710)

### Model Training 
1. Train PocketNet with ArcFace loss
   + ./train.sh

2. Train PocketNet with template knowledge distillation
    + ./train_kd.sh
3. Train PocketNet with multi-step template knowledge distillation
    + ./train_kd.sh
 
### To-do 
- [ ] Add pretrained model
- [ ] Add search code
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