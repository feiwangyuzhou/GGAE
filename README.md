# GGAE: Global Graph Attention Embedding Network for Relation Prediction in Knowledge Graphs

Source code for our [TNNLS] paper: [Global Graph Attention Embedding Network for Relation Prediction in Knowledge Graphs]

### Requirements
- [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

Please download miniconda from above link and create an environment using the following command:

        conda env create -f pytorch35.yml

Activate the environment before executing the program as follows:

        source activate pytorch35
### Dataset
We used three different datasets for evaluating our model. All the datasets and their folder names are given below.
- Freebase: FB15k-237
- Wordnet: WN18RR
- Kinship: kinship


### Reproducing results

To reproduce the results published in the paper:

        $ bash train_xx.sh


### Citation
Please cite our paper if you use this code in your work.

@ARTICLE{9452178,
  author={Li, Qian and Wang, Daling and Feng, Shi and Niu, Cheng and Zhang, Yifei},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Global Graph Attention Embedding Network for Relation Prediction in Knowledge Graphs}, 
  year={2021},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2021.3083259}}


For any clarification, comments, or suggestions please create an issue or contact feiwangyuzhou@163.com
