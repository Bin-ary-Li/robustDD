import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet50
import pytorch_lightning as pl
import numpy as np
import torchmetrics
import argparse
import os

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

def make_cnn(c=64, num_classes=10):
    ''' Returns a 5-layer CNN with width parameter c. '''
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.BatchNorm2d(c),
        nn.ReLU(),

        # Layer 1
        nn.Conv2d(c, c*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*2),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(c*2, c*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*4),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(c*4, c*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*8),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c*8, num_classes, bias=True)
    )

# define the LightningModule
class LitCNN(pl.LightningModule):
    def __init__(self, cnn, lr=1e-4):
        super().__init__()
        self.cnn = cnn
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, labels = batch
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = self.cnn(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        _, preds = torch.max(outputs.data, 1)
        acc = torchmetrics.functional.accuracy(preds, labels)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, labels = batch
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = self.cnn(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        _, preds = torch.max(outputs.data, 1)
        acc = torchmetrics.functional.accuracy(preds, labels)

        # Logging to TensorBoard by default
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def generate_noisy_label(labels, p, n_cls):
    noisy_targets = labels.copy()
    for i, target in enumerate(labels):
        if np.random.random_sample() <= p:
            incorrect_labels = [i for i in range(n_cls) if i != target]
            np.random.shuffle(incorrect_labels)
            noisy_targets[i] = incorrect_labels[0]
    return noisy_targets

def doArgs():
    parser = argparse.ArgumentParser(description='parameters for Double Decent')
    parser.add_argument('--dataset', type=str, help="the name of the dataset to use, currently support one of these strings: [cifar10]", required=True)
    parser.add_argument('--noise', type=float, help='making the training label noisy, 0 means no noise', default=0)
    parser.add_argument('--savepath', type=str, help="save path for log and model checkpoint, default to ./logs", default='logs')
    parser.add_argument('--cls', type=int, help="num of classes", required=True)
    parser.add_argument('--width', type=int, help="width of model", required=True)
    parser.add_argument('--epoch', type=int, help="max number of epoch to run, default 10", default=10)
    parser.add_argument('--resumeckpt', type=str, help='path of model checkpoint to resume', default='')
    parser.add_argument('--loadnoisydata', type=str, help='load train data that has noisy label (to guarantee reproducibility when resume training)', default='')
    return parser.parse_args()

# model hyperparameter 
def main():
    args = doArgs()

    lr = 0.0001
    batch_size = 256
    
    max_epoch = args.epoch
    width = args.width
    n_cls = args.cls
    dataset = args.dataset.lower()
    label_noise = args.noise
    resume_ckpt = args.resumeckpt
    noisy_data_path = args.loadnoisydata
    save_path = args.savepath

    # mod = resnet50(width_per_group=width)
    # mod.fc = torch.nn.Linear(2048, n_cls)
    mod = make_cnn(width, n_cls)
    litmod = LitCNN(mod, lr=lr)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if (dataset == 'cifar10'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
    else:
        return

    logger = pl.loggers.CSVLogger(save_path, name=f"cnn_width{width}_noise{label_noise}")

    # check if want to resume training
    if (resume_ckpt != ''):
        # if load noisy data for resuming training
        if label_noise > 0:
            assert noisy_data_path != ''
            trainset = torch.load(noisy_data_path)
    else:
        # If generating noisy data for the first time, save it to the logging directory
        if (noisy_data_path == '' and label_noise > 0):
            noisy_labels = generate_noisy_label(trainset.targets, label_noise, n_cls)
            trainset.targets = noisy_labels
            os.makedirs(logger.root_dir)
            torch.save(trainset, os.path.join(logger.root_dir, 'noisy_train.pt'))


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator="auto",
        logger=logger,
        devices=1)
    
    if (resume_ckpt == ''):
        trainer.fit(litmod, trainloader, testloader)
    else:
        trainer.fit(litmod, trainloader, testloader, ckpt_path=resume_ckpt)


if __name__ == '__main__':
    main()
