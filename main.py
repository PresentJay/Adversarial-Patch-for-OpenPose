import torch
import os
from src import configs, datasets, models, networks, patches


if __name__ == '__main__':
    
    # load the Network Settings
    # args styles from Pytorch Examples - https://github.com/pytorch/examples/blob/master/imagenet/main.py
    args = configs.init_args()
    
    # set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    
    # set the model
    global netClassifier
    netClassifier = models.getModels_fromTV(args.netClassifier)
    print(f'attack model = {netClassifier.__class__}')
    
    # load the dataset
    # TODO: apply statusbar
    print("prepare dataset from =>", args.data_dir)
    train_loader, test_loader = datasets.load_data(args)
    print('dataset is loaded. . . complete')

    # measure the accuracy about dataset
    trainset_acc, test_acc = networks.test(netClassifier, train_loader), networks.test(netClassifier, test_loader)
    print(f'Accuracy of the model on clean trainset and testset is {trainset_acc}% and {test_acc}%')

    # initialize patch
    patch = patches.init_patch(args)
    
    # TODO: apply statusbar
    # train patch
    patches.train_patch(args, train_loader, test_loader, patch, netClassifier)
    
    