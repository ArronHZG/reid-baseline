import numpy as np
import torch
import visdom
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from sklearn.cluster import DBSCAN
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from engine.uda import compute_dist


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        import torch.nn.functional as F

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=train_batch_size, shuffle=True
    )

    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False), batch_size=val_batch_size, shuffle=False
    )
    return train_loader, val_loader


def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))


def run(train_batch_size, val_batch_size, epochs, lr, log_interval):
    device = "cuda"
    model = Net().cuda()
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    loss = CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "nll": Loss(loss)}, device=device)

    checkpointer = ModelCheckpoint("mnist-cluster",
                                   "resnet50",
                                   n_saved=1,
                                   require_empty=False)

    vis = visdom.Visdom()
    train_loss_window = create_plot_window(vis, "#Iterations", "Loss", "Training Loss")
    train_avg_loss_window = create_plot_window(vis, "#Iterations", "Loss", "Training Average Loss")
    train_avg_accuracy_window = create_plot_window(vis, "#Iterations", "Accuracy", "Training Average Accuracy")
    val_avg_loss_window = create_plot_window(vis, "#Epochs", "Loss", "Validation Average Loss")
    val_avg_accuracy_window = create_plot_window(vis, "#Epochs", "Accuracy", "Validation Average Accuracy")

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}] " +
              f"Iteration[{engine.state.iteration}/{len(train_loader)}] Loss: {engine.state.output:.2f}")
        vis.line(
            X=np.array([engine.state.iteration]),
            Y=np.array([engine.state.output]),
            update="append",
            win=train_loss_window,
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )
        vis.line(
            X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]), win=train_avg_accuracy_window, update="append"
        )
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_nll]), win=train_avg_loss_window, update="append")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )
        vis.line(
            X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]), win=val_avg_accuracy_window, update="append"
        )
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_nll]), win=val_avg_loss_window, update="append")

        checkpointer(engine, {"model": model})

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)


def create_cluster(dist: torch.Tensor):
    """
    If want to use DBSCAN, we need to affirm the best epsilon in DBSCAN.
    :param dist:
    :return:
    """
    rho = 1.6e-3
    dist = dist.triu(1)  # the upper triangular part of a matrix
    dist = dist.view(dist.size(0) ** 2, -1).squeeze()
    dist = dist[dist.nonzero()].squeeze()  # get all distance  dim = 1
    sorted_dist, _ = dist.sort()
    top_num = torch.tensor(rho * sorted_dist.size()[0]).round().to(torch.int)
    if top_num <= 20:
        top_num = 20
    print(top_num)
    eps = sorted_dist[:top_num].mean().to(torch.int).cpu().numpy()
    # logger.info(f'eps in cluster: {eps:.3f}')
    print(eps)
    return DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)


something = None


def generate_self_label(dist_matrix):
    global something
    if something is None:
        print("create_cluster")
        something = create_cluster(dist_matrix)

    dist = dist_matrix.cpu().numpy()
    print(type(dist))

    labels = something.fit_predict(dist)
    num_ids = len(set(labels)) - 1
    return num_ids, labels


def cluster(train_batch_size, val_batch_size):
    device = "cuda"
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    checkpointer = ModelCheckpoint("mnist-cluster",
                                   "resnet50",
                                   n_saved=1,
                                   require_empty=False)
    checkpoint = torch.load("mnist-cluster/resnet50_model_4690.pth")
    checkpointer.load_objects({'model': model}, checkpoint)
    model.cuda()
    model.eval()

    # evaluator = create_supervised_evaluator(
    #     model, metrics={"accuracy": Accuracy()}, device=device)
    # evaluator.run(val_loader)
    # metrics = evaluator.state.metrics
    # print(f"Validation Results Avg accuracy: {metrics['accuracy']:.2f}")
    features = []
    labels = []
    for image, label in val_loader:
        with torch.no_grad():
            image = image.cuda()
            label = label.cuda()
            feature = model(image)
            features.append(feature)
            labels.append(label)

    features = torch.cat(features)
    target = torch.cat(labels)

    dict_matrix = compute_dist(features, if_re_ranking=False)
    class_num, labels = generate_self_label(dict_matrix)
    print(f"class_num {class_num}")
    for i in range(target.size()[0]):
        print(f"{target[i]}  :  {labels[i]}")


if __name__ == "__main__":
    # run(64, 64, 5, 0.01, 100)
    cluster(100, 10)
