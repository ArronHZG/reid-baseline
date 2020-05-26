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

| backbone                  | id    | tricks                                                       | mAP  | rank1 | mean  |
| ------------------------- | ----- | ------------------------------------------------------------ | ---- | ----- | ----- |
| resnet50                  | 01/02 | baseline                                                     | 85.8 | 94.2  | 90.10 |
| resnet50                  | 03    | batch-size: 128                                              | 85.0 | 93.8  | 89.97 |
| resnet50                  | 04    | loss: softmax_arcface_triplet                                | 70.9 | 89.2  | 81.01 |
| resnet50                  | 05    | METRIC_LOSS_WEIGHT: 2.0                                      | 86.5 | 94.1  | 90.30 |
| resnet50                  | 07    | METRIC_LOSS_WEIGHT: 10.0                                     | 84.4 | 92.6  | 88.50 |
| resnet50                  | 08    | METRIC_LOSS_WEIGHT: 5                                        | 85.6 | 93.9  | 89.70 |
| resnet50                  | 09    | learning weight                                              | 86.7 | 93.9  | 90.36 |
| se_resnet50               | 11    | merge resnet50: 09                                           | 86.2 | 93.6  | 89.94 |
| ibn_a_resnet50            | 12    | merge resnet50: 09                                           | 88.3 | 95.0  | 91.68 |
| ibn_b_resnet50            | 13    | merge resnet50: 09                                           | 83.6 | 92.8  | 88.17 |
| se_ibn_a_resnet50         | 14    | merge resnet50: 09                                           | 87.9 | 94.5  | 91.21 |
| se_ibn_b_resnet50         | 15    | merge resnet50: 09                                           | 82.6 | 92.6  | 87.63 |
| resnet50                  | 16    | merge resnet50: 05+09                                        | 86.5 | 93.9  | 90.18 |
| resnet50                  | 17    | merge resnet50: 16 <br/>$loss = a_p + (a_p - a_n + mergin)_+$ | 86.2 | 94.3  | 90.26 |
| resnet101                 | 01    | learning weight                                              | 87.7 | 94.6  | 91.15 |
| se_resnet101              | 02    | merge resnet101: 01                                          | 87.5 | 94.3  | 90.98 |
| ibn_a_resnet101           | 03    | merge resnet101: 01                                          | 88.6 | 95    | 91.77 |
| ibn_b_resnet101           | 04    | merge resnet101: 01                                          | 83.6 | 92.8  | 88.22 |
| se_ibn_a_resnet101        | 05    | merge resnet101: 01                                          | 88.6 | 94.8  | 91.70 |
| se_ibn_b_resnet101        | 06    | merge resnet101: 01                                          | 83.7 | 93.5  | 88.57 |
| resnext101_32x8d          | 01    | learning weight                                              | 88.6 | 95.0  | 91.81 |
| se_resnext101_32x8d       | 02    | merge resnext101_32x8d: 01                                   | 88.3 | 95.0  | 91.66 |
| ibn_a_resnext101_32x8d    | 03    | merge resnext101_32x8d: 01                                   | 89.0 | 95.3  | 92.16 |
| ibn_b_resnext101_32x8d    | 04    | merge resnext101_32x8d: 01                                   | 84.1 | 93.1  | 88.61 |
| se_ibn_a_resnext101_32x8d | 05    | merge resnext101_32x8d: 01                                   | 88.6 | 95.3  | 91.98 |
| se_ibn_b_resnext101_32x8d | 06    | merge resnext101_32x8d: 01                                   | 83.4 | 92.7  | 88.29 |
|                           |       |                                                              |      |       |       |
|                           |       |                                                              |      |       |       |
|                           |       |                                                              |      |       |       |
#### test

using rerank

| backbone               | id   | mAP  | rank1 | mean  |
| ---------------------- | ---- | ---- | ----- | ----- |
| resnet50               | 02   | 94.2 | 95.0  | 94.6  |
| ibn_a_resnet50         | 12   | 94.4 | 95.6  | 94.97 |
| ibn_a_resnet101        | 03   | 94.5 | 95.8  | 95.17 |
| ibn_a_resnext101_32x8d | 03   | 94.5 | 95.7  | 95.11 |

### DukeMTMC-reID

#### train

| backbone       | id   | tricks          | mAP  | rank1 | mean  |
| -------------- | ---- | --------------- | ---- | ----- | ----- |
| resnet50       | 01   |                 | 76.4 | 86.7  | 81.58 |
| ibn_a_resnet50 | 02   | learning weight | 75.4 | 86.0  | 80.65 |
| ibn_a_resnet50 | 03   | learning weight | 76.8 | 87.9  | 82.33 |

### msmt17

#### train
| backbone       | id   | tricks          | mAP  | rank1 | mean |
| -------------- | ---- | --------------- | ---- | ----- | ---- |
| resnet50       | 01   | learning weight | 46.2 | 64.8  | 55.6 |
| ibn_a_resnet50 | 02   | learning weight | 51.4 | 69.8  | 60.6 |
|                |      |                 |      |       |      |
## Join train

### market1501--DukeMTMC--msmt17 

#### baseline and target
| backbone       | id   | tricks                        | market1501<br>map/rank-1 | dukemtmc<br/>map/rank-1 | msmt17<br/>map/rank-1 |
| -------------- | ---- | ----------------------------- | ------------------------ | ----------------------- | --------------------- |
| resnet50       | 01   | Epoch: 520                    | 86.8/94.6                | 78.6/88.0               | 48.9/67.5             |
| ibn_a_resnet50 | 02   | learning weight<br>Epoch: 540 |                          |                         |                       |
|                |      |                               |                          |                         |                       |

## Continual learning

### market1501--DukeMTMC

#### Learning without Forgetting (LwF)

- LOSS_TYPE: softmax_triplet
- ID_LOSS_WEIGHT: 1.0
- BASE_LR: 0.000035

| backbone | id   | tricks                                          | market<br>map/forget | duck<br>map |
| -------- | ---- | ----------------------------------------------- | -------------------- | ----------- |
| ibn_b_50 | 01   | ce_dist:10                                      | 47.0/-36.3           | 66.3        |
| ibn_b_50 | 02   | merge ibn_b_resnet50: 01<br>ID_LOSS_WEIGHT: 0.2 | 46.7/-36.6           | 65.0        |
| ibn_b_50 | 03   | tr_dist                                         | 48.6/-34.7           | 66.6        |
| ibn_b_50 | 04   | ce_dist:10 tr_dist                              | 48.6/-34.7           | 66.6        |
| ibn_b_50 | 05   | LOSS_TYPE: triplet<br>30 epoch                  | 47.1/-36.2           | 57.2        |
| ibn_a_50 |      | tr_dist                                         |                      |             |
| ibn_a_50 |      | ce_dist:10                                      |                      |             |
| ibn_a_50 |      | ce_dist:10 tr_dist                              |                      |             |
|          |      |                                                 |                      |             |

#### Encoder Based Lifelong Learning (EBLL)

##### after GAP

first-step: training market with autoencoder

| backbone | id   | tricks                          | ae-loss                                        | market<br/>map | market<br/>rank-1 |
| -------- | ---- | ------------------------------- | ---------------------------------------------- | -------------- | ----------------- |
| ibn_a_50 | 18   | CODE_SIZE: 1024<br>LAMBDA: 0.01 | ae: 0.0010<br>ae_l1: 0.0194<br/>ae_l2: 0.0014  | 88             | 95                |
| ibn_a_50 | 19   | CODE_SIZE: 512<br>LAMBDA: 0.01  | ae: 0.0010<br>ae_l1: 0.0104<br/>ae_l2: 0.0014  | 88.1           | 94.9              |
| ibn_a_50 | 20   | CODE_SIZE: 1024<br>LAMBDA: 0.1  | ae: 0.0029<br/>ae_l1: 0.6125<br/>ae_l2: 0.1796 | 87.7           | 94.7              |
| ibn_a_50 | 21   | CODE_SIZE: 1024<br>LAMBDA: 0.05 | ae: 0.0010<br/>ae_l1: 0.0915<br/>ae_l2: 0.0014 | 88             | 94.6              |

second-step: training duck, using ibn_a_50: 19 weight.

```python
CONTINUATION.LOSS_TYPE = 'tr_dist'
```

| backbone | id   | tricks                   | market<br/>map/forget | duck<br/>map |
| -------- | ---- | ------------------------ | --------------------- | ------------ |
| ibn_a_50 | 06   | ae_bce_loss              | 42                    | 72.8         |
| ibn_a_50 | 07   | ae_l1_bce_loss           | 41.8                  | 72.9         |
| ibn_a_50 | 08   | ae_l2_bce_loss           | 41.6                  | 73.0         |
| ibn_a_50 | 09   | bce: ae_loss ae_l1 ae_l2 | 41.8                  | 72.7         |
| ibn_a_50 | 10   | ae_mse_loss              | 41.9                  | 72.9         |
| ibn_a_50 | 11   | ae_l1_mse_loss           | 42.4                  | 72.7         |
| ibn_a_50 | 12   | ae_l2_mse_loss           | 42.9                  | 72.8         |
| ibn_a_50 | 13   | mse: ae_loss ae_l1 ae_l2 | 43.2                  | 72.6         |

##### before GAP

# Acknowledgements

## person-reid

**Bag of Tricks and A Strong ReID Baseline**

[michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)

**Unsupervised Deep Embedding for Clustering Analysis**

[arxiv](https://arxiv.org/abs/1511.06335)
[DeepClustering](https://github.com/Deepayan137/DeepClustering)

**A Discriminative Feature Learning Approach for Deep Face Recognition**

[centerloss](https://github.com/jxgu1016/MNIST_center_loss_pytorch/blob/master/MNIST_with_centerloss.py)

## unsupervised

**Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification(SSG)**

[OasisYang/SSG](https://github.com/OasisYang/SSG)
[visualizing-dbscan-clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

## continual learning

**Learning without Forgetting**

**Encoder Based Lifelong Learning**

[Pytorch-implementation-of-Encoder-Based-Lifelong-learning](https://github.com/rahafaljundi/Pytorch-implementation-of-Encoder-Based-Lifelong-learning)

