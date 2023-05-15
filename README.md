

# UDA_CLR
Data and code for paper titled [Unsupervised Domain Adaptive Fundus Image Segmentation with Category-Level Regularization](https://arxiv.org/abs/2207.03684) (MICCAI 2022 paper)

Existing unsupervised domain adaptation methods based on adversarial learning have achieved good performance in several medical imaging tasks. However, these methods focus only on global distribution adaptation and ignore distribution constraints at the category level, which would lead to sub-optimal adaptation performance. This paper presents an unsupervised domain adaptation framework based on category-level regularization that regularizes the category distribution from three perspectives. Specifically, for inter-domain category regularization, an adaptive prototype alignment module is proposed to align feature prototypes of the same category in the source and target domains. In addition, for intra-domain category regularization, we tailored a regularization technique for the source and target domains, respectively. In the source domain, a prototype-guided discriminative loss is proposed to learn more discriminative feature representations by enforcing intra-class compactness and inter-class separability, and as a complement to traditional supervised loss. In the target domain, an augmented consistency category regularization loss is proposed to force the model to produce consistent predictions for augmented/unaugmented target images, which encourages semantically similar regions to be given the same label.


## Contents
[1. Data](#data)

[2. Model](#model)

[3. Requirements](#requirements)

[4. Running](#running)

[5. Citation](#citation)

## Data
In order to train and test the model, you first need to download the [Drishti-GS](https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation),
 [RIM-ONE](http://medimrg.webs.ull.es/research/downloads/) and [refuge](https://refuge.grand-challenge.org/) datasets and place them in the folder path ' ./data '.
## Model
You can train deeplab v3+ using mobilenetv2 or others as backbone.

## Requirements
* python==3.8
* pytorch==1.11.0
* scipy==1.8.0
* numpy==1.21.6
* scikit-learn==1.1.1
* tensorboardX==1.4 
* matplotlib 
* pillow
* pyyaml

## Running
Training and testing our model through the bash scripts:
```
CUDA_VISIBLE_DEVICES=0 python train_use_fix_initial.py
```
You can also add or change parameters in train_use_fix_initial.py

## Citation
If our paper or code is helpful to you, please consider citing our paper:
```
@inproceedings{feng2022unsupervised,
  title={Unsupervised Domain Adaptive Fundus Image Segmentation with Category-Level Regularization},
  author={Feng, Wei and Wang, Lin and Ju, Lie and Zhao, Xin and Wang, Xin and Shi, Xiaoyu and Ge, Zongyuan},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2022: 25th International Conference, Singapore, September 18--22, 2022, Proceedings, Part II},
  pages={497--506},
  year={2022},
  organization={Springer}
}
```
