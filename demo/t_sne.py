# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率


def main():
    data = np.load('../engine/qf_feat.npy')
    label = np.load('../engine/qf.npy')
    # digits = datasets.load_digits(n_class=10)
    # data = digits.data
    # label = digits.target

    # tsne = TSNE(n_components=2, verbose=1)
    # data = tsne.fit_transform(data)
    # np.save('qf_tsne.npy', data)
    data = np.load('qf_tsne.npy')

    print(data.shape)
    print(label.shape)
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    plt.scatter(data[:, 0], data[:, 1], c=label, s=1)

    nums = set()

    for i in range(label.shape[0]):
        if label[i] not in nums:
            nums.add(label[i])
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 size=1.0)

    print(len(nums))

    plt.xticks([])
    plt.yticks([])

    plt.savefig("test.png")
    plt.savefig("test.svg", format='svg')
    plt.show()


if __name__ == '__main__':
    main()
