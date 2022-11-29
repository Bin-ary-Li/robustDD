import os
import argparse

def doArgs():
    parser = argparse.ArgumentParser(description='run 4 trainings at a time')

    parser.add_argument('--path', type=str, help="python file path, default='./resnet50.py'", default='./resnet50.py')
    parser.add_argument('--dataset', type=str, help="dataset path, or type Cifar10 or Cifar100 to download dataset", required=True)
    parser.add_argument('--cls', type=int, help="num of classes", required=True)
    parser.add_argument('--startwidth', type=int, help="start width of model, will cover it and the following 3 width", required=True)
    parser.add_argument('--savepath', type=str, help="save path for log and model checkpoint, default to ./logs", default='logs')
    parser.add_argument('--epoch', type=int, help="max number of epoch to run, default 10", default=10)
    return parser.parse_args()

def main():
    args = doArgs()

    path = args.path

    width = args.startwidth

    n_cls = args.cls

    dataset = args.dataset

    savepath = args.savepath

    epoch = args.epoch 
    
    for i in range(2):
        os.system('python {} --dataset {} --cls {} --width {} --savepath {} --epoch {}'.format(path, dataset, n_cls, width+i, savepath, epoch))


if __name__ == '__main__':
    main()