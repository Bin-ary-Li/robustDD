from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader
import torchvision.transforms as transforms
from torchvision.models import DenseNet
import torch
import pytorch_lightning as pl
import argparse

# define the LightningModule
class LitDensenet(pl.LightningModule):
    ''' Returns a Densenet121 with growth parameter k. '''
    def __init__(self, trainset, testsets, testset_names, num_workers=2, 
                k=32, num_classes=10, lr=1e-4, train_batch_size=256, test_batch_size=256):
        super().__init__()
        self.num_workers = num_workers
        self.trainset = trainset
        self.testsets = testsets
        self.testset_names = testset_names
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.model = DenseNet(growth_rate=k, num_classes=num_classes)
        self.lr = lr

    def train_dataloader(self):
        train_loader=  get_train_loader("standard", self.trainset, 
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

    def training_step(self, batch, batch_idx):
        inputs, labels, metadata = batch
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        _, y_pred = torch.max(outputs.data, 1)
        return {'loss':loss,
                'y_pred':y_pred.cpu(),
                'labels':labels.cpu(),
                'metadata':metadata.cpu()}
    
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
            else:
                self.log(key,value, on_epoch=True)


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
                else:
                    self.log(key,value)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# def generate_noisy_label(labels, p, n_cls):
#     noisy_targets = labels.copy()
#     for i, target in enumerate(labels):
#         if np.random.random_sample() <= p:
#             incorrect_labels = [i for i in range(n_cls) if i != target]
#             np.random.shuffle(incorrect_labels)
#             noisy_targets[i] = incorrect_labels[0]
#     return noisy_targets

def doArgs():
    parser = argparse.ArgumentParser(description='parameters for Double Decent')
    parser.add_argument('--dataset', type=str, help="the name of the dataset to use, currently support one of these strings: [camelyon17]", default='camelyon17')
    parser.add_argument('--noise', type=float, help='making the training label noisy, NO SUPPORTED for now!', default=0)
    parser.add_argument('--savepath', type=str, help="save path for log and model checkpoint, default to ./logs", default='logs')
    parser.add_argument('--cls', type=int, help="num of classes", required=True)
    parser.add_argument('--hparam', type=int, help="hparameter of model complexity, control densenet growth rate", required=True)
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
    resume_ckpt = args.resumeckpt
    noisy_data_path = args.loadnoisydata
    save_path = args.savepath

    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset=dataset, download=True)

    trans = transforms.Compose(
            [transforms.ToTensor()]
        )

    train_data = dataset.get_subset(
        "train",
        frac=1,
        transform=trans,
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

    model = LitDensenet(train_data, test_datasets, test_split_names, num_worker=num_worker,
                        k=growth_rate, num_classes=n_cls, lr=lr, train_batch_size=batch_size,
                        test_batch_size=batch_size)

    logger = pl.loggers.CSVLogger('logs', 
                            name=f"densenet_width{growth_rate}_noise{label_noise}")

    # # check if want to resume training
    # if (resume_ckpt != ''):
    #     # if load noisy data for resuming training
    #     if label_noise > 0:
    #         assert noisy_data_path != ''
    #         trainset = torch.load(noisy_data_path)
    # else:
    #     # If generating noisy data for the first time, save it to the logging directory
    #     if (noisy_data_path == '' and label_noise > 0):
    #         noisy_labels = generate_noisy_label(trainset.targets, label_noise, n_cls)
    #         trainset.targets = noisy_labels
    #         os.makedirs(logger.root_dir)
    #         torch.save(trainset, os.path.join(logger.root_dir, 'noisy_train.pt'))


    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                     shuffle=True, num_workers=2)

    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                     shuffle=False, num_workers=2)

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator="gpu",
        precision=16,
        logger=logger,
        devices=1,)
    
    # if (resume_ckpt == ''):
    #     trainer.fit(litmod, trainloader, testloader)
    # else:
    #     trainer.fit(litmod, trainloader, testloader, ckpt_path=resume_ckpt)

    trainer.fit(model)


if __name__ == '__main__':
    main()
