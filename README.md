# PocketNet



## This is the official repository of the paper:
#### PocketNet: Extreme Lightweight Face Recognition Network using Neural Architecture Search and  Multi-Step Knowledge Distillation

![evaluation](https://raw.githubusercontent.com/fdbtrs/PocketNet/main/logs/tradeoff.png)


### Face recognition  model training 
Model training:
In the paper, we employ MS1MV2 as the training dataset which can be downloaded from InsightFace (MS1M-ArcFace in DataZoo)
Download [MS1MV2](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view?usp=sharing) dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) on strictly follow the licence distribution

Unzip the dataset and place it in the data folder

Rename the config/config_xxxxxx.py to config/config.py
1. Train PocketNet with ArcFace loss
   + ./train.sh
2. Train PocketNet with template knowledge distillation
    + ./train_kd.sh
3. Train PocketNet with multi-step template knowledge distillation
    + ./train_kd.sh

| Model  | Parameters (M)| configuration | log| pretrained model| 
| ------------- | ------------- |  ------------- |------------- |------------- |
| PocketNetS-128 |0.92 |[Config](config/config_PocketNetS128.py)| [log](https://drive.google.com/file/d/1NVdir7oRO_JN7Pci_7PQLnUbLw-0rIkX/view?usp=sharing)|[Pretrained-model](https://drive.google.com/drive/folders/1ziq5l_gPjlz6zs7CWq7ae_KhHFhMcS0h?usp=sharing)  |
| PocketNetS-256 |0.99 |[Config](config/config_PocketNetS256.py)| [log](https://drive.google.com/file/d/1kjAWj1QuwyQEqllOL3MC4IOZ1Y0iujNc/view?usp=sharing)|[Pretrained-model](https://drive.google.com/drive/folders/1rOrPBE5M9c8Hx1hJCUCjflbbtRVfY3q0?usp=sharing) |
| PocketNetM-128 |1.68 |[Config](config/config_PocketNetM128.py) | [log](https://drive.google.com/file/d/1zC3xjGhf7dax_JW0tIYAionH9dHjrByp/view?usp=sharing)|[Pretrained-model](https://drive.google.com/drive/folders/1p1OUoIKHWv0it5wb9xaFX-ZvZkqPiu7O?usp=sharing)  |
| PocketNetM-256 |1.75 |[Config](config/config_PocketNetM256.py)| [log](https://drive.google.com/file/d/1uTMnddbZEniaZGS8b9_bKbSZ1zeVzsH7/view?usp=sharing) |[Pretrained-model](https://drive.google.com/drive/folders/1VFt_Eq04iEIoB7krhgWd70IkBI3XEs3y?usp=sharing)  |



Intall the requirement from requirement.txt

pip install -r requirements.txt

All code are trained and tested using PyTorch 1.7.1

Details are under (Torch)[https://pytorch.org/get-started/locally/]



## Face recognition evaluation
##### evaluation on LFW, AgeDb-30, CPLFW, CALFW and CFP-FP: 
1. download the data from their offical webpages.
2. alternative: The evaluation datasets are available in the training dataset package as bin file
3. Rename the configuration file in config directory based on the evaluation model e.g. rename config_PocketNetM128.py to config.py to evaluate the PocketNetM128
4. set the config.rec to dataset folder e.g. data/faces_emore
5. set the config.val_targets for list of the evaluation dataset
6. download the pretrained model from link the previous table
7. set the config.output to path to pretrained model weights
8. run eval/eval.py
9. the output is test.log contains the evaluation results over all epochs

##### evaluation on IJB-B and IJB-C: 

1. Please apply for permissions from NIST before your usage [NIST_Request](https://nigos.nist.gov/datasets/ijbc/request)
2. run eval/IJB/runIJBEval.sh
 

### Differentiable architecture search training
The code of NAS is available under NAS
### To-do 
- [x] Add pretrained model
- [x] Training configuration
- [x] Add NAS code
- [x] Add evaluation results

 
