from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics import loss
from wilds.common.utils import get_counts
import torchvision.transforms as transforms
from torchvision.models import DenseNet
import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
import numpy as np
import pandas as pd
import os

# Adapted from https://github.com/YBZh/Bridging_UDA_SSL

from PIL import Image, ImageOps, ImageEnhance, ImageDraw


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    if v < 0:
        return img
    w, h = img.size
    x_center = _sample_uniform(0, w)
    y_center = _sample_uniform(0, h)

    x0 = int(max(0, x_center - v / 2.0))
    y0 = int(max(0, y_center - v / 2.0))
    x1 = min(w, x0 + v) 
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


FIX_MATCH_AUGMENTATION_POOL = [
    (AutoContrast, 0, 1),
    (Brightness, 0.05, 0.95),
    (Color, 0.05, 0.95),
    (Contrast, 0.05, 0.95),
    (Equalize, 0, 1),
    (Identity, 0, 1),
    (Posterize, 4, 8),
    (Rotate, -30, 30),
    (Sharpness, 0.05, 0.95),
    (ShearX, -0.3, 0.3),
    (ShearY, -0.3, 0.3),
    (Solarize, 0, 256),
    (TranslateX, -0.3, 0.3),
    (TranslateY, -0.3, 0.3),
]


def _sample_uniform(a, b):
    return torch.empty(1).uniform_(a, b).item()


class RandAugment:
    def __init__(self, n, augmentation_pool):
        assert n >= 1, "RandAugment N has to be a value greater than or equal to 1."
        self.n = n
        self.augmentation_pool = augmentation_pool

    def __call__(self, img):
        ops = [
            self.augmentation_pool[torch.randint(len(self.augmentation_pool), (1,))]
            for _ in range(self.n)
        ]
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * _sample_uniform(0, 1)
            img = op(img, val)
        cutout_val = _sample_uniform(0, 1) * 0.5
        img = Cutout(img, cutout_val)
        return img

class Flatten(nn.Module):
    def forward(self, x): 
        return x.view(x.size(0), x.size(1))

def make_cnn(c=64, num_classes=10):
    ''' Returns a 6-layer CNN with width parameter c. '''
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
        nn.Conv2d(c*8, c*16, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 5
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c*16, num_classes, bias=True)
    )

class GroupDRO(pl.LightningModule):
    def __init__(self, trainset, testsets, testset_names, loss, grouper, is_group_in_train,
                group_dro_step_size = 0.01, model = 'densenet', num_workers=6,
                k=32, num_classes=10, lr=1e-4, train_batch_size=256, test_batch_size=256):
        super().__init__()
        self.trainset = trainset
        self.testsets = testsets
        self.testset_names = testset_names
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.model_name = model
        self.num_workers = num_workers
        if self.model_name == 'densenet':
            self.model = DenseNet(growth_rate=k, num_classes=num_classes)
        else:
            self.model = make_cnn(c=k, num_classes=num_classes)
        self.lr = lr

        self.loss = loss
        self.grouper = grouper
        # step size
        self.group_weights_step_size = group_dro_step_size
        # initialize adversarial weights
        self.group_weights = torch.zeros(grouper.n_groups)
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
        self.group_weights = self.group_weights.cuda()

    def train_dataloader(self):
        train_loader=  get_train_loader("group", self.trainset, 
                                grouper=self.grouper, n_groups_per_batch=2,
                                uniform_over_groups=True, distinct_groups=True,
                                batch_size=self.train_batch_size,
                                num_workers=self.num_workers, pin_memory=True)
        return train_loader
    
    def val_dataloader(self):
        val_loaders = []
        for dataset in self.testsets:
            loader = get_eval_loader("standard", dataset, 
                                batch_size=self.test_batch_size,
                                num_workers=self.num_workers, pin_memory=True)
            val_loaders.append(loader)
        return val_loaders

    def process_batch(self, batch):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch 
        """
        x, y_true, metadata = batch
        x = x.cuda()
        y_true = y_true.cuda()
        g = self.grouper.metadata_to_group(metadata.cpu()).cuda()

        outputs = self.model(x)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'group_weight': self.group_weights
        }

        return results

    def compute_objective_update_group(self, results):
        """
        Takes an output of SingleModelAlgorithm.process_batch() and computes the
        optimized objective. For group DRO, the objective is the weighted average
        of losses, where groups have weights groupDRO.group_weights.
        Args:
            - results (dictionary): output of SingleModelAlgorithm.process_batch()
        Output:
            - objective (Tensor): optimized objective; size (1,).
        """
        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.grouper.n_groups,
            return_dict=False)

        # update group weights
        self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*group_losses.data)
        self.group_weights = (self.group_weights/(self.group_weights.sum()))
        # save updated group weights
        results['group_weight'] = self.group_weights

        objective = group_losses @ self.group_weights

        return objective

    def training_step(self, batch, batch_idx):
        results = self.process_batch(batch)
        loss = self.compute_objective_update_group(results)

        _, y_pred = torch.max(results['y_pred'], 1)
        return {'loss':loss,
                'y_pred':y_pred.cpu(),
                'labels':results['y_true'].cpu(),
                'metadata':results['metadata'].cpu()}
    
    def training_epoch_end(self, train_step_outputs):
        preds = [x["y_pred"] for x in train_step_outputs]
        labels = [x["labels"] for x in train_step_outputs]
        metadata = [x["metadata"] for x in train_step_outputs]
        eval = self.trainset.eval(torch.cat(preds), 
                            torch.cat(labels), 
                            torch.cat(metadata))
        for key, value in eval[0].items():
            if 'avg' in key or 'wg' in key:
                self.log(key+'_'+'train', value, on_epoch=True)
            # else:
            #     self.log(key,value, on_epoch=True)
        for i, w in enumerate(self.group_weights):
            self.log(f'group_{i}', w, on_epoch=True)


    def validation_step(self, batch, batch_idx, dataloader_idx):
        inputs, labels, metadata = batch
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        _, y_pred = torch.max(outputs.data, 1)
        return {'loss':loss,
                'test_idx':dataloader_idx,
                'y_pred':y_pred.cpu(),
                'labels':labels.cpu(),
                'metadata':metadata.cpu()}

    def validation_epoch_end(self, val_step_outputs) -> None:
        for k in range(len(self.testsets)):
            preds = []
            labels = []
            metadata = []
            for x in val_step_outputs[k]:
                preds.append(x['y_pred'])
                labels.append(x['labels'])
                metadata.append(x['metadata'])
            eval = self.testsets[k].eval(torch.cat(preds), 
                                        torch.cat(labels), 
                                        torch.cat(metadata))
            # Logging
            testset_name = self.testset_names[k]
            for key, value in eval[0].items():
                if 'avg' in key or 'wg' in key:
                    self.log(key+'_'+testset_name, value)
                # else:
                #     self.log(key,value)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def add_noise_to_metadata(csv_path, p):
    # add label noise to train data
    metadata = pd.read_csv(csv_path, 
                            index_col=0,
                            dtype={'patient': 'str'})
    y_arr = metadata[metadata['split'] == 0]['tumor'].to_numpy() # training split
    flip_loc = np.random.choice([True,False], len(y_arr), p=[p, 1-p])
    noisy_y = np.logical_not(y_arr, where = flip_loc, out=y_arr.copy())
    metadata.loc[metadata['split']==0, 'tumor'] = noisy_y
    metadata.to_csv(csv_path)

def doArgs():
    parser = argparse.ArgumentParser(description='parameters for Double Decent')
    parser.add_argument('--randaugment', type=int, help="add random augmentation to training data, 1 means add augmentation", default=0)
    parser.add_argument('--dataset', type=str, help="the name of the dataset to use, currently support one of these strings: [camelyon17]", default='camelyon17')
    parser.add_argument('--noise', type=float, help='making the training label noisy, NO SUPPORTED for now!', default=0)
    parser.add_argument('--savepath', type=str, help="save path for log and model checkpoint, default to ./logs", default='logs')
    parser.add_argument('--cls', type=int, help="num of classes", required=True)
    parser.add_argument('--model', type=str, help="which model to use, support: densenet, cnn", default='densenet')
    parser.add_argument('--hparam', type=int, help="hparameter of model complexity, e.g. control densenet growth rate", required=True)
    parser.add_argument('--epoch', type=int, help="max number of epoch to run, default 10", default=10)
    parser.add_argument('--resumeckpt', type=str, help='path of model checkpoint to resume', default='')
    parser.add_argument('--worker', type=int, help="number of workers for dataloader, more workers need more memory, default 2", default=2)
    parser.add_argument('--loadnoisydata', type=str, help='load train data that has noisy label (to guarantee reproducibility when resume training)', default='')
    return parser.parse_args()

# model hyperparameter 
def main():
    args = doArgs()

    lr = 0.0001
    batch_size = 256
    
    max_epoch = args.epoch
    growth_rate = args.hparam
    n_cls = args.cls
    dataset = args.dataset.lower()
    label_noise = args.noise
    num_worker = args.worker
    is_augment = args.randaugment
    resume_ckpt = args.resumeckpt
    # noisy_data_path = args.loadnoisydata
    save_path = args.savepath
    model_name = args.model

    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset=dataset, download=True)

    # If adding noise to the training data
    if label_noise > 0:
        abspath = os.path.abspath('.')
        noisy_data_path = os.path.join(abspath, f'noisy_data_{label_noise}')
        if not os.path.exists(noisy_data_path):
            print('Noisy dataset not found, creating noisy dataset...')
            os.mkdir(noisy_data_path)
            os.system(f'cp -R {abspath}/data/camelyon17_v1.0 {noisy_data_path}/camelyon17_v1.0')
        metadata_file = os.path.join(noisy_data_path, 'camelyon17_v1.0/metadata.csv')
        add_noise_to_metadata(metadata_file, label_noise)
        dataset = get_dataset(dataset="camelyon17", download=False, root_dir=noisy_data_path)

    if is_augment == 1:
        train_trans = transforms.Compose(
                [
                RandAugment(2, FIX_MATCH_AUGMENTATION_POOL),
                transforms.ToTensor()]
            )
    else:
        train_trans = transforms.Compose(
            [
            transforms.ToTensor()]
        )

    trans = transforms.Compose(
            [
            transforms.ToTensor()]
        )

    train_data = dataset.get_subset(
        "train",
        frac=1,
        transform=train_trans,
    )

    # Get the test set
    test_data = dataset.get_subset(
        "test",
        transform=trans
    )

    val_data = dataset.get_subset(
        'val',
        transform=trans
    )

    id_val_data = dataset.get_subset(
        'id_val',
        transform=trans
    )

    test_datasets = [test_data, val_data, id_val_data]
    test_split_names = ['test', 'val', 'idval']

    grouper = CombinatorialGrouper(dataset, ['hospital'])
    train_g = grouper.metadata_to_group(train_data.metadata_array)
    is_group_in_train = get_counts(train_g, grouper.n_groups) > 0
    ce_loss = loss.Loss(torch.nn.functional.cross_entropy)

    model = GroupDRO(train_data, test_datasets, test_split_names, num_workers=num_worker,
                    loss=ce_loss, grouper=grouper, is_group_in_train=is_group_in_train,
                        k=growth_rate, num_classes=n_cls, lr=lr, train_batch_size=batch_size,
                        test_batch_size=batch_size, model=model_name)

    logger = pl.loggers.CSVLogger('logs', 
                            name=f"groupdro_complexity{growth_rate}_noise{label_noise}_randaug{is_augment}_{model_name}")


    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator="gpu",
        precision=16,
        logger=logger,
        devices=1,)
    
    trainer.fit(model)


if __name__ == '__main__':
    main()
