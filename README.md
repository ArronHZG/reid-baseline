## recurrent

### Bag of Tricks and A Strong ReID Baseline

[michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)

### Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification(SSG)

[OasisYang/SSG](https://github.com/OasisYang/SSG)
[visualizing-dbscan-clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

### Unsupervised Deep Embedding for Clustering Analysis

[arxiv](https://arxiv.org/abs/1511.06335)
[DeepClustering](https://github.com/Deepayan137/DeepClustering)


### Center Loss

[centerloss](https://github.com/jxgu1016/MNIST_center_loss_pytorch/blob/master/MNIST_with_centerloss.py)

## ecosystem

[ignite](https://github.com/pytorch/ignite)

[tensorboardX](https://github.com/lanpa/tensorboardX)

UNDO [dali](https://github.com/NVIDIA/DALI)

UNDO [jpeg4py](https://github.com/ajkxyz/jpeg4py)

[apex](https://github.com/NVIDIA/apex)


```shell script

sudo apt-get install libturbojpeg

```

## parameter

## market1501

### original strong-baseline
```shell script
Validation Results - Epoch: 120
mAP: 85.8%
CMC curve, Rank-1  :94.2%
CMC curve, Rank-5  :98.3%
CMC curve, Rank-10 :98.9%
```
### baseline (resnet50 experiment-1)
```shell script
Validation Results - Epoch: 120
mAP: 85.9%
CMC curve, Rank-1  :94.2%
CMC curve, Rank-5  :98.4%
CMC curve, Rank-10 :99.0%
```
### apex O1 (resnet50 experiment-2)
```shell script
Validation Results - Epoch: 120
mAP: 85.8%
CMC curve, Rank-1  :94.2%
CMC curve, Rank-5  :98.2%
CMC curve, Rank-10 :99.0%
```
### batch-size 128 (resnet50 experiment-3)
```shell script
Validation Results - Epoch: 120
mAP: 85.0%
CMC curve, Rank-1  :93.6%
CMC curve, Rank-5  :98.0%
CMC curve, Rank-10 :98.8%
```
### apex O2 (resnet50 experiment-4)
```shell script
Validation Results - Epoch: 120
mAP: 85.9%
CMC curve, Rank-1  :94.2%
CMC curve, Rank-5  :98.2%
CMC curve, Rank-10 :99.0%

Validation Results - Epoch: 120
mAP: 86.2%
CMC curve, Rank-1  :94.1%
CMC curve, Rank-5  :98.2%
CMC curve, Rank-10 :98.8%

test-rerank
Validation Results
mAP: 94.2%
CMC curve, Rank-1  :95.7%
CMC curve, Rank-5  :98.0%
CMC curve, Rank-10 :98.6%

test-after
Validation Results
mAP: 85.6%
CMC curve, Rank-1  :94.1%
CMC curve, Rank-5  :97.7%
CMC curve, Rank-10 :98.6%

test-after-rerank
Validation Results
mAP: 93.4%
CMC curve, Rank-1  :95.0%
CMC curve, Rank-5  :97.4%
CMC curve, Rank-10 :98.0%

```
### baseline (resnet101 experiment-1)
```shell script
Validation Results - Epoch: 120
mAP: 87.1%
CMC curve, Rank-1  :94.3%
CMC curve, Rank-5  :98.4%
CMC curve, Rank-10 :99.2%
```
### baseline (se_resnet50 experiment-1)
```
mAP: 86.4%
CMC curve, Rank-1  :94.4%
CMC curve, Rank-5  :98.4%
CMC curve, Rank-10 :99.2%
```
### baseline (se_resnet101 experiment-1)
```shell script
The test feature is normalized
Validation Results - Epoch: 120
mAP: 87.1%
CMC curve, Rank-1  :94.2%
CMC curve, Rank-5  :98.2%
CMC curve, Rank-10 :99.0%
Save best
```
### baseline (se_resnext50_32x4d experiment-1) 1h30m
```shell script
Validation Results - Epoch: 120
mAP: 87.8%
CMC curve, Rank-1  :95.0%
CMC curve, Rank-5  :98.5%
CMC curve, Rank-10 :99.1%
```
### baseline (se_resnext101_32x4d experiment-1) 2h20m
```shell script
Validation Results - Epoch: 120
mAP: 87.9%
CMC curve, Rank-1  :94.7%
CMC curve, Rank-5  :98.5%
CMC curve, Rank-10 :99.0%
```
### baseline (resnet50_ibn_a experiment-1)
```shell script
Validation Results - Epoch: 120
mAP: 86.9%
CMC curve, Rank-1  :94.8%
CMC curve, Rank-5  :98.5%
CMC curve, Rank-10 :99.1%
```
### baseline (resnet101_ibn_a experiment-1)
```shell script
Validation Results - Epoch: 120
mAP: 87.6%
CMC curve, Rank-1  :94.9%
CMC curve, Rank-5  :98.5%
CMC curve, Rank-10 :99.0%
```
## DukeMTMC-reID

### baseline
```shell script
mAP: 76.4%
CMC curve, Rank-1  :86.7%
CMC curve, Rank-5  :94.1%
CMC curve, Rank-10 :96.1%
```

## msmt17

### baseline
```shell script
mAP: 45.8%
CMC curve, Rank-1  :64.0%
CMC curve, Rank-5  :76.6%
CMC curve, Rank-10 :81.0%
```

## market-->duke
```shell script
times-5
Validation Results
mAP: 44.8%
CMC curve, Rank-1  :51.8%
CMC curve, Rank-5  :59.8%
CMC curve, Rank-10 :65.7%
```


## market1501--DukeMTMC-reID--msmt17 

### baseline and target
```shell script
--------------------------------------------------------------------------------
('/home/arron/dataset/market1501', 3368) Validation Results - Epoch: 120
mAP: 85.1%
CMC curve, Rank-1  :93.6%
CMC curve, Rank-5  :98.0%
CMC curve, Rank-10 :98.8%
('/home/arron/dataset/DukeMTMC-reID', 2228) Validation Results - Epoch: 120
mAP: 77.3%
CMC curve, Rank-1  :86.8%
CMC curve, Rank-5  :94.0%
CMC curve, Rank-10 :95.7%
('/home/arron/dataset/msmt17', 11659) Validation Results - Epoch: 120
mAP: 47.0%
CMC curve, Rank-1  :64.9%
CMC curve, Rank-5  :77.1%
CMC curve, Rank-10 :81.5%
Save best: 0.7579
--------------------------------------------------------------------------------
```