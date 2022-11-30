import os
import argparse

def doArgs():
    parser = argparse.ArgumentParser(description='run 4 trainings at a time')

    parser.add_argument('--path', type=str, help="python file path, default='./resnet50.py'", default='cnn_train.py')
    parser.add_argument('--dataset', type=str, help="name of the dataset to use, support: cifar10", default='cifar10')
    parser.add_argument('--cls', type=int, help="num of classes", required=True)
    parser.add_argument('--width', type=str, help="width of cnn model to run, if pass multiple one use comma to seperate", required=True)
    parser.add_argument('--savepath', type=str, help="save path for log and model checkpoint, default to ./logs", default='logs')
    parser.add_argument('--epoch', type=int, help="max number of epoch to run, default 10", default=10)
    parser.add_argument('--noise', type=float, help="add label noise, 0 means no noise and 1 is all noise", default=0)
    return parser.parse_args()

def main():
    args = doArgs()

    path = args.path

    width = [int(item) for item in args.width.split(',')]

    n_cls = args.cls

    dataset = args.dataset

    savepath = args.savepath

    epoch = args.epoch 
    
    noise = args.noise

    for i in width:
        command = f'python {path} --dataset {dataset} --cls {n_cls} --width {i} --savepath {savepath} --epoch {epoch} --noise {noise}'
        os.system(command)


if __name__ == '__main__':
    main()