


### To-do 
- [x] Update configuration file
- [x] Update augment_cnn


Unzipped the  dataset and place it in the data folder
The aligned and cropped version of CASIA-WebFace can be downloaded it from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)

The database are aligned based on the util/align_trans.py
Set search configurations in util/config.py, and path to dataset folder (config.root).
In utils/config.py, specify the name of the search with config.name
1. Run search algorithm search.py
   + ./search.sh
2. Go to searchs/{config.name}/ to see the result of the search


## Reference
[https://github.com/quark0/darts](https://github.com/quark0/darts) (official implementation)
[https://github.com/khanrc/pt.darts](https://github.com/khanrc/pt.darts)

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
```
@inproceedings{DBLP:conf/iclr/LiuSY19,
  author    = {Hanxiao Liu and
               Karen Simonyan and
               Yiming Yang},
  title     = {{DARTS:} Differentiable Architecture Search},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  publisher = {OpenReview.net},
  year      = {2019},
  url       = {https://openreview.net/forum?id=S1eYHoC5FX},
  timestamp = {Thu, 25 Jul 2019 14:25:55 +0200},
  biburl    = {https://dblp.org/rec/conf/iclr/LiuSY19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```