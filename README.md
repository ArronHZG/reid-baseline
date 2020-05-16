# recurrent

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



# ecosystem

[ignite](https://github.com/pytorch/ignite)

[tensorboardX](https://github.com/lanpa/tensorboardX)

UNDO [dali](https://github.com/NVIDIA/DALI)

UNDO [jpeg4py](https://github.com/ajkxyz/jpeg4py)

[apex](https://github.com/NVIDIA/apex)

# parameter

# experiment

## direct

baseline:

- batch-size: 64
- instance: 4
- loss: softmax_triplet
- METRIC_LOSS_WEIGHT:1

### market1501

#### train

| backbone                  | id    | tricks                                                   | mAP  | rank1 | mean  |
| ------------------------- | ----- | -------------------------------------------------------- | ---- | ----- | ----- |
| resnet50                  | 01/02 | baseline                                                 | 85.8 | 94.2  | 90.10 |
| resnet50                  | 03    | batch-size: 128                                          | 85.0 | 93.8  | 89.97 |
| resnet50                  | 04    | loss: softmax_arcface_triplet                            | 70.9 | 89.2  | 81.01 |
| resnet50                  | 05    | METRIC_LOSS_WEIGHT: 2.0                                  | 86.5 | 94.1  | 90.30 |
| resnet50                  | 07    | METRIC_LOSS_WEIGHT: 10.0                                 | 84.4 | 92.6  | 88.50 |
| resnet50                  | 08    | METRIC_LOSS_WEIGHT: 5                                    | 85.6 | 93.9  | 89.70 |
| resnet50                  | 09    | learning weight                                          | 86.7 | 93.9  | 90.36 |
| se_resnet50               | 11    | merge resnet50: 09                                       | 86.2 | 93.6  | 89.94 |
| ibn_a_resnet50            | 12    | merge resnet50: 09                                       | 88.3 | 95.0  | 91.68 |
| ibn_b_resnet50            | 13    | merge resnet50: 09                                       | 83.6 | 92.8  | 88.17 |
| se_ibn_a_resnet50         | 14    | merge resnet50: 09                                       | 87.9 | 94.5  | 91.21 |
| se_ibn_b_resnet50         | 15    | merge resnet50: 09                                       | 82.6 | 92.6  | 87.63 |
| resnet50                  | 16    | merge resnet50: 05+09                                    | 86.5 | 93.9  | 90.18 |
| resnet50                  | 17    | merge resnet50: 16 $loss = a_p + (a_p - a_n + mergin)_+$ | 86.2 | 94.3  | 90.26 |
| resnet101                 | 01    | learning weight                                          | 97.7 | 94.6  | 91.15 |
| se_resnet101              | 02    | merge resnet101: 01                                      | 87.5 | 94.3  | 90.98 |
| ibn_a_resnet101           | 03    | merge resnet101: 01                                      | 88.6 | 95    | 91.77 |
| ibn_b_resnet101           | 04    | merge resnet101: 01                                      | 83.6 | 92.8  | 88.22 |
| se_ibn_a_resnet101        | 05    | merge resnet101: 01                                      | 88.6 | 94.8  | 91.70 |
| se_ibn_b_resnet101        | 06    | merge resnet101: 01                                      | 83.7 | 93.5  | 88.57 |
| resnext101_32x8d          | 01    | merge resnext101_32x8d: 01                               | 88.6 | 95.0  | 91.81 |
| se_resnext101_32x8d       | 02    | merge resnext101_32x8d: 01                               | 88.3 | 95.0  | 91.66 |
| ibn_a_resnext101_32x8d    | 03    | merge resnext101_32x8d: 01                               | 89.0 | 95.3  | 92.16 |
| ibn_b_resnext101_32x8d    | 04    | merge resnext101_32x8d: 01                               | 84.1 | 93.1  | 88.61 |
| se_ibn_a_resnext101_32x8d | 05    | merge resnext101_32x8d: 01                               | 88.6 | 95.3  | 91.98 |
| se_ibn_b_resnext101_32x8d | 06    | merge resnext101_32x8d: 01                               | 83.4 | 92.7  | 88.29 |
|                           |       |                                                          |      |       |       |
|                           |       |                                                          |      |       |       |
|                           |       |                                                          |      |       |       |

#### test

using rerank

| backbone               | id   | mAP  | rank1 | mean  |
| ---------------------- | ---- | ---- | ----- | ----- |
| resnet50               | 02   | 94.2 | 95.0  | 94.6  |
| ibn_a_resnet50         | 12   | 94.4 | 95.6  | 94.97 |
| ibn_a_resnet101        | 03   | 94.5 | 95.8  | 95.17 |
| ibn_a_resnext101_32x8d | 03   | 94.5 | 95.7  | 95.11 |

### DukeMTMC-reID

#### baseline
```shell script
mAP: 76.4%
CMC curve, Rank-1  :86.7%
CMC curve, Rank-5  :94.1%
CMC curve, Rank-10 :96.1%
```

### msmt17

#### baseline
```shell script
mAP: 45.8%
CMC curve, Rank-1  :64.0%
CMC curve, Rank-5  :76.6%
CMC curve, Rank-10 :81.0%
```

## Join train

### market1501--DukeMTMC-reID--msmt17 

#### baseline and target
```shell script
--------------------------------------------------------------------------------
 Valid - Epoch: 400
 market1501 Validation Results
 mAP: 86.8%
 CMC curve, Rank-1  :94.6%
 CMC curve, Rank-5  :98.2%
 CMC curve, Rank-10 :98.8%
 dukemtmc Validation Results
 mAP: 78.6%
 CMC curve, Rank-1  :88.0%
 CMC curve, Rank-5  :94.6%
 CMC curve, Rank-10 :96.2%
 msmt17 Validation Results
 mAP: 48.9%
 CMC curve, Rank-1  :67.5%
 CMC curve, Rank-5  :79.2%
 CMC curve, Rank-10 :83.1%
 Save best: 0.7740
 --------------------------------------------------------------------------------
```

