# 【ReID】首届“全国人工智能大赛”（行人重识别 Person ReID 赛项）

[首届“全国人工智能大赛”（行人重识别 Person ReID 赛项](https://www.kesci.com/home/competition/5d90401cd8fc4f002da8e7be/content)

## 比赛的内容

行人重识别（Person Re-identification）是利用计算机视觉技术判断图像或者视频序列中是否存在特定行人的技术。 目前，行人重识别已经成为学术界的研究热点。行人重识别研究可以有效推动计算机视觉算法对相似物体的区分能力，提升对物体类内差异的鲁棒性。

本赛道分初赛、复赛、决赛三个阶段，每个阶段的任务为基于图片的行人重识别：即给定一张含有某个行人的查询图片，行人重识别算法需要在行人图像库中查找并返回特定数量的含有该行人的图片。比赛将使用首位准确率（Rank-1 Accuracy）和mAP（mean Average Precision）作为客观性能指标。

**预训练模型要求**

1. 不允许使用任何含有行人图像/行人身体部件的外部数据集，或外部数据集上的预训练模型.

2. 只允许使用在 ImageNet 数据上训练的标准图像分类模型，用于辅助 ReID 模型初始化.

3. 不允许选手在 ImageNet 图像数据上自行训练深度模型。
4. 不允许选手对大赛数据集进行额外标注

**初赛数据集**

训练集中含有 4,768 个行人，20,429 张图片，每张图片提供 ID 标签，可以用于模型训练

测试集包含 6,714 张

**复赛数据集**

训练集含有 9,968 个行人，85,729 张 png 格式图片，提供 ID 标签，可以用于模型训练

测试集第一阶段测试集包含 71,460 张, 第二阶段包含165,975 张 

## **methodology**

### DataSet

**数据集分析**

1. 本次比赛的数据集数据量巨大, 对比market1501和dukemtmc, ID数量和图片数量都增加了10倍. 大的数据集可以保证泛化性(但是由于原因4, 泛化性并没有提高), 但是也会使得单次训练时间过长.

2. 对于公开数据集, 基本上进行了清洗和整理, 在训练集中同一个ID对应的图片数量基本一致.但是比赛数据集存在着长尾效应, 一个ID对应图片最多可能上千张,最少可能1到2张.

3. 由于比赛对于个人隐私的保密性,以及为了提高难度, 数据集经过特殊处理基本上肉眼无法辨识是否为同一人. 在后期的研究观察中发现, 每一张图片是灰度图(0-255)在三个区间进行分离重新组合为RGB彩色图.

   <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gjeqrbmxufj31dy08s42l.jpg" alt="image-20200414192626811" style="zoom:50%;" />

   实际的训练集:列出一些较为

   <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gjeqrctwvfj30ue096mz6.jpg" alt="image-20200414190444329" style="zoom:50%;" />

   在训练中, 大量出现上图所示的图片, 人眼基本不无法辨认, 只能靠机器识别:dog:.另外官方也声明了不允许去使用某些算法还原训练集.

   并且通过对于训练集观察, 同一ID的颜色模式基本一直, 不同ID会存在变化.

4. 训练集和测试集的分布不同, 所以是一个UDA问题^[1]^
   > <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gjeqreh0qej30wy0lw77q.jpg" alt="image-20200414193506890" style="zoom:50%;" />

   

使用随机擦除, 随机翻转, 随机crop,  增加了颜色通道的增强.

长尾问题: 减少ID过于多的图片

在比赛将近结尾的时候, 发现了问题4, 使用了SSG^[2]^对测试集进行挖掘, 但是提交的时候没有用, 太遗憾了.

### Backbone

backbone由于题目限制, 只能使用公开的ImageNet训练后的模型, 所以基础实验使用ResNet50, 集合实验使用ResNet101-IBN-a, ResNet101-IBN-b

如果没有限制, 使用NAS搜索该数据最优结构应该是效果最好的.

### Special Head

global feature 使用 strong baseline^[13]^ 的版本

part feature 在初赛使用PCB^[3]^作为输出头, 复赛使用MGN^[4]^作为输出头.

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gjeqrdsaskj317u0h8n3s.jpg" alt="image-20200515154627588" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gjeqrcdiw2j318o0smajn.jpg" alt="image-20200414202315894" style="zoom:50%;" />

增加了ABD-net^[12]^的attention部分, 但是明显的提升.

### Loss

在 strong baseline^[5]^ 中, 使用了soft cross entropy loss作为ID loss, triplet loss 作为 metric loss, 同时使用center loss 增强效果.

比赛中使用过 arcface loss^[6]^, rank loss^[7]^ 替代 cross entropy 作为ID loss, 但是效果不大, 等[1]开源之后, 可以看看超参

并且考虑在训练过程中调参 修改 ID loss 和 metric loss 的权重.

在40epoch 将 ID loss : metric loss 设为 1:2

### Learning Strategies

Look Ahead + RAdam, github 有一个实现好了的[ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer).

### Train stricks

使用[apex](https://github.com/NVIDIA/apex)混合精度训练

### Test stricks

1. 使用了rerank^[8]^, 在罗浩的 [1]版本中在公开数据集没有问题, 但是在比赛数据集中, 由于gallery太大, 复赛中distance metrix 是一个$17,000\times17,000$的矩阵, 再加上各种运算, 内存爆了若干次, 后期改成离线保存版本.
2. 使用Baseline + PCB + MGN 进行离线模型融合

### Problem

1. 通过参考[1]的方案分享, 发现最大的问题就在于没有很好地解决UDA, 使用了SSG但是没有很好地调优
2. 并且尝试了不同颜色模式使用不同的模型转变为multi-task任务, 在红蓝图上效果显著提升, 绿紫图对比平均水平下降, 但是最终由于缺乏红蓝图绿紫图的类别标注, 无法设计任务判定模块, 最后也弃用了
3. rerank 在提交时, 内存爆掉 (我的锅😭)
4. loss 有了思路, 但是调参是个技术活, [一文道尽softmax loss及其变种](https://zhuanlan.zhihu.com/p/34044634)
5. 每天只有一次提交机会, 正确的验证集真的很重要, 可能有一些trick是有效的, 但是线下降点, 线上就没有验证.

### Methodology Of Other Teams

[1] 的方案

*等大神开源代码, 占个位置*

使用类似SSG的做法, 对target domain进行挖掘, 挖掘过后的生成pseudo label, rerank 后与训练集合并迭代训练.

 1. 没有对全部训练集挖掘, 而是通过对图像进行均值和方差统计, 分为A域和C域, 红蓝域数据直接预测, 紫域数据进行挖掘

 2. 迭代训练时, 更换backbone, backbone的复杂程度依次上升Resnet101-ibn-a → Resnet10-ibn-b → SeResnet-ibn-a, 因为模型越复杂越容易过拟合, 而且前期挖掘数据并不准确, 在逐渐挖掘的同时, 提升模型提取特征的能力, 这个点特别好.

    <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gjeqrdaf0jj312m0aawfy.jpg" alt="image-20200414230618826" style="zoom:50%;" />

 3. 挖掘的时候, 并没有采用传统意义上的无监督挖掘,而是进行一个半监督挖掘. 无监督挖掘例如DBSCAN会根据feature之间距离的远近自动生成聚类的 cluster center, 聚类的数量和实际结果不一致.而[1]使用Query作为中心预先决定了cluster center, 之后再聚类, 效果肯定要优于无监督挖掘. 这个其实已经不是聚类了, 直观来讲就是一个排序.

 4. 决赛方案暂且略过.

[anxiangsir](https://github.com/anxiangsir)/**[NAIC_reid_challenge_rank14_rank25](https://github.com/anxiangsir/NAIC_reid_challenge_rank14_rank25)**

1. 线下使用4折交叉验证
2. 使用 RandomPatch
3. 方差反转
4. 魔改MGN
5. 图片尺寸增加, 配合MGN

[haoni0812](https://github.com/haoni0812)/**[person-reid-2019-NAIC](https://github.com/haoni0812/person-reid-2019-NAIC)**

1. 使用了A-softmax-loss
2. 魔改MGN
3. 平衡同一图片ID数量

[maliho0803](https://github.com/maliho0803)/**[NAIC_reid_challenge](https://github.com/maliho0803/NAIC_reid_challenge)**

1. 使用 rank loss
2. MGN + PCB
3. 随机crop

[bochuanwu](https://github.com/bochuanwu)/**[NAIC_Person_Reid](https://github.com/bochuanwu/NAIC_Person_Reid)**

1. rank loss
2. 修改 rank loss 为 聚类 loss
3. OSMLoss, 加权对比损失函数
4. backbone 使用 Pyramidal^[10]^ , 一个更为复杂的part feature model
5. 增加了attention机制 DANet^[11]^, 和ABD-net^[12]^思路一样.

[lxyICT](https://github.com/lxyICT)/**[ReID-json-rel](https://github.com/lxyICT/ReID-json-rel)**

## Reference

[1]: 罗浩团队分享 https://zhuanlan.zhihu.com/p/109920765
[2]: Self-similarity Grouping A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification
[3]: Beyond Part Models: Person Retrievalwith Refined Part Pooling(and A Strong Convolutional Baseline)
[4]: Learning Discriminative Features with Multiple Granularities for Person Re-Identification
[5]: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
[6]: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
[7]: Ranked List Loss for Deep Metric Learning
[8]:Re-ranking Person Re-identification with k-reciprocal Encoding
[9]:Deep Metric Learning by Online Soft Mining and Class-Aware Attention
[10]:Pyramidal Person Re-IDentification via Multi-Loss Dynamic Training
[11]:Dual Attention Network for Scene Segmentation
[12]:ABD-Net: Attentive but Diverse Person Re-Identification
[13]: Bag of Tricks and A Strong Baseline for Deep Person Re-identification