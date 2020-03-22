#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import visdom
import numpy as np
from PIL import Image  # type: ignore
import base64 as b64  # type: ignore
from io import BytesIO
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import decomposition, manifold
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
mnist_dataset = MNIST(download=True, root=".", transform=data_transform, train=True)
import seaborn as sns

sns.set(style="darkgrid")

sub_sample = 400
images = []
labels = []
for index, sample in enumerate(mnist_dataset):
    if index == sub_sample:
        break
    image, label = sample
    image = image.view(1, -1)
    images.append(image)
    labels.append(labels)

features = torch.cat(images)
print(features.shape)

vis = visdom.Visdom()

image_datas = []
for feat in features:
    img_array = np.flipud(np.rot90(np.reshape(feat, (28, 28))))
    im = Image.fromarray(img_array * 255)
    im = im.convert('RGB')
    buf = BytesIO()
    im.save(buf, format='PNG')
    b64encoded = b64.b64encode(buf.getvalue()).decode('utf-8')
    image_datas.append(b64encoded)


def get_mnist_for_index(id):
    image_data = image_datas[id]
    display_data = 'data:image/png;base64,' + image_data
    return "<img src='" + display_data + "' />"


vis.embeddings(features, labels, data_getter=get_mnist_for_index, data_type='html')

input('Waiting for callbacks, press enter to quit.')
