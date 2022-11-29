import os
import argparse

def doArgs():
    parser = argparse.ArgumentParser(description='run 4 trainings at a time')

    parser.add_argument('--path', type=str, help="python file path, default='./resnet50.py'", default='./resnet50.py')
    parser.add_argument('--dataset', type=str, help="dataset path, or type Cifar10 or Cifar100 to download dataset", required=True)
    parser.add_argument('--cls', type=int, help="num of classes", required=True)
    parser.add_argument('--startwidth', type=int, help="start width of model, will cover it and the following 3 width", required=True)
    return parser.parse_args()

def main():
    args = doArgs()

    path = args.path

    width = args.startwidth

    n_cls = args.cls

    dataset = args.dataset

    for i in range(4):
        os.system('python {} --dataset {} --cls {} --width {}'.format(path, dataset, n_cls, width+i))


if __name__ == '__main__':
    main()