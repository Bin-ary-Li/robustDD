import os
import argparse

def doArgs():
    parser = argparse.ArgumentParser(description='run 4 trainings at a time')

    parser.add_argument('--path', type=str, help="python file path", default='wilds_camelyon_train.py')
    parser.add_argument('--dataset', type=str, help="name of the dataset to use, support: camelyon17", default='camelyon17')
    parser.add_argument('--cls', type=int, help="num of classes", default=2)
    parser.add_argument('--randaugment', type=int, help="add random augmentation to training data, 1 means add augmentation", default=0)
    parser.add_argument('--hparam', type=str, help="hparameter of model complexity, control densenet growth rate, if pass multiple one use comma to seperate", required=True)
    parser.add_argument('--savepath', type=str, help="save path for log and model checkpoint, default to ./logs", default='logs')
    parser.add_argument('--epoch', type=int, help="max number of epoch to run, default 10", default=10)
    parser.add_argument('--model', type=str, help="which model to use, support: densenet, cnn", default='densenet')
    parser.add_argument('--noise', type=float, help="add label noise, 0 means no noise and 1 is all noise", default=0)
    parser.add_argument('--worker', type=int, help="number of worker", default=2)
    return parser.parse_args()

def main():
    args = doArgs()

    path = args.path

    hparam = [int(item) for item in args.hparam.split(',')]

    n_cls = args.cls

    dataset = args.dataset

    savepath = args.savepath

    epoch = args.epoch 
    
    noise = args.noise

    worker = args.worker

    for i in hparam:
        command = f'python {path} --dataset {dataset} --cls {n_cls} --hparam {i} --savepath {savepath} --epoch {epoch} --noise {noise} --worker {worker} --randaugment {args.randaugment} --model {args.model}'
        os.system(command)


if __name__ == '__main__':
    main()