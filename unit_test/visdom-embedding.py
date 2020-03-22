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

images = torch.cat(images).numpy()
print(images.shape)
y = labels
X = images

n_samples, n_features = X.shape
n_neighbors = 30


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X_emb, title=None):
    x_min, x_max = np.min(X_emb, 0), np.max(X_emb, 0)
    X_emb = (X_emb - x_min) / (x_max - x_min)

    for i in range(X_emb.shape[0]):
        if i % 10 == 0:
            print(i)
        plt.text(X_emb[i, 0], X_emb[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    # print(offsetbox)
    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(sub_sample):
    #         dist = np.sum((X_emb[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 8e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X_emb[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(X[i].reshape(28, 28)[::2, ::2], cmap=plt.cm.gray_r),
    #             X_emb[i])
    #         ax.add_artist(imagebox)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# ----------------------------------------------------------------------
def plot_digits():
    # Plot images of the digits
    n_img_per_row = 20
    img = np.zeros((30 * n_img_per_row, 30 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 30 * i + 1
        for j in range(n_img_per_row):
            iy = 30 * j + 1
            img[ix:ix + 28, iy:iy + 28] = X[i * n_img_per_row + j].reshape((28, 28))

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('A selection from the 64-dimensional digits dataset')
    plt.show()
    plt.cla()


plot_digits()
# t-SNE embedding of the digits dataset
print("SVD")
X_pca = decomposition.TruncatedSVD(n_components=50).fit_transform(X)
print("Computing t-SNE embedding")
X_tsne = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)

print(X_tsne.shape)

# plot_embedding(X_tsne)
sns.relplot(data=X, col=y)
