import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from ignite.engine import Engine, Events
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

# EMBEDDING VISUALIZATION FOR A TWO-CLASSES PROBLEM

# just a bunch of loss

if torch.cuda.is_available():
    device = "cuda"


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.cn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.cn2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(in_features=800, out_features=10)

    def forward(self, i):
        i = self.cn1(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        i = self.cn2(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        i = i.view(len(i), -1)
        i = self.fc1(i)
        i = F.log_softmax(i, dim=1)
        return i


# dataset
train_sets = datasets.MNIST("~/dataset",
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())

val_sets = datasets.MNIST("~/dataset",
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())

train_loader = DataLoader(train_sets, batch_size=100)
val_loader = DataLoader(val_sets, batch_size=100)

# network
model = M()

# loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

trainer = create_supervised_trainer(model, optimizer, loss, device=device)
evaluator = create_supervised_evaluator(model,
                                        metrics={
                                            'accuracy': Accuracy(),
                                            'loss': Loss(loss)})


@trainer.on(Events.ITERATION_COMPLETED(every=50))
def log_training_loss(engine):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))


trainer.run(train_loader, max_epochs=100)

#
# # settings for train and log
# num_epochs = 5
# embedding_log = 5
# writer = SummaryWriter(comment='mnist_embedding_training')
#
# # TRAIN
# for epoch in range(num_epochs):
#     for j, sample in enumerate(gen):
#         n_iter = (epoch * len(gen)) + j
#         # reset grad
#         m.zero_grad()
#         optimizer.zero_grad()
#         # get batch data
#         data_batch = Variable(sample[0], requires_grad=True).float()
#         label_batch = Variable(sample[1], requires_grad=False).long()
#         # FORWARD
#         out = m(data_batch)
#         loss_value = loss(out, label_batch)
#         # BACKWARD
#         loss_value.backward()
#         optimizer.step()
#         # LOGGING
#         writer.add_scalar('loss', loss_value.data.item(), n_iter)
#
#         if j % embedding_log == 0:
#             # we need 3 dimension for tensor to visualize it!
#             out = torch.cat((out.data, torch.ones(len(out), 1)), 1)
#             writer.add_embedding(
#                 out,
#                 metadata=label_batch.data,
#                 label_img=data_batch.data,
#                 global_step=n_iter)
#
#     print(loss_value.item())
# writer.close()
#
#
# # tensorboard --logdir runs
#
#
# def update_model(trainer, batch):
#     model.train()
#     optimizer.zero_grad()
#     x, y = prepare_batch(batch)
#     y_pred = model(x)
#     loss = loss_fn(y_pred, y)
#     loss.backward()
#     optimizer.step()
#     return loss.item()
#
#
# trainer = Engine(update_model)
#
#
# @trainer.on(Events.STARTED)
# def on_training_started(engine):
#     print("Another message of start training")
#
#
# @trainer.on(Events.ITERATION_COMPLETED(every=50))
# def log_training_loss_every_50_iterations(engine):
#     print("{} / {} : {} - loss: {:.2f}"
#           .format(engine.state.epoch, engine.state.max_epochs, engine.state.iteration, engine.state.output))
#
#
# trainer.run(data, max_epochs=100)
