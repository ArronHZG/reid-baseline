import torch
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver, ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"


class Saver:
    def __init__(self):
        self.best_result = 0

    def save(self, s):
        self.best_result = s
        np.save("best.npy", self.best_result)

    def resume(self):
        self.best_result = np.load("best.npy")


# loading the saved model
def fetch_last_checkpoint_model_filename(model_save_path, name):
    import os
    checkpoint_files = os.listdir(model_save_path)
    checkpoint_files = [f for f in checkpoint_files if '.pth' in f and name in f]
    checkpoint_iter = [
        int(x.split('_')[2].split('.')[0])
        for x in checkpoint_files]
    last_idx = np.array(checkpoint_iter).argmax()
    return os.path.join(model_save_path, checkpoint_files[last_idx])


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

train_loader = DataLoader(train_sets, batch_size=50)
val_loader = DataLoader(val_sets, batch_size=50)

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


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))


to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}

checkpointer = ModelCheckpoint('./log/models/a',
                               'train',
                               n_saved=1,
                               require_empty=False)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, to_save)

best_pointer = ModelCheckpoint('./log/models/b',
                               'best',
                               n_saved=1,
                               require_empty=False)
evaluator.best_result = 0

saver = Saver()


@trainer.on(Events.EPOCH_COMPLETED, checkpointer=best_pointer, to_save=to_save, saver=saver)
def log_validation_results(trainer, checkpointer, to_save, saver):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))

    if saver.best_result < evaluator.state.metrics['accuracy']:
        print('save best')
        saver.save(evaluator.state.metrics['accuracy'])
        checkpointer(trainer, to_save)


evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, to_save)

path = fetch_last_checkpoint_model_filename('./log/models/a', 'train')
print(path)
pointer = torch.load(path)
checkpointer.load_objects(to_load=to_save, checkpoint=pointer)
trainer.state.max_epochs = 10
saver.resume()
print(saver.best_result)

trainer.run(train_loader, max_epochs=10)