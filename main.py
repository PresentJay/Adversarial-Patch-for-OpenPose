import torch
import os
import sys
from src import configs, datasets, models, patches


if __name__ == '__main__':
    
    # load the Network Settings
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
    
    
    trainset_acc = models.test(netClassifier, train_loader, cuda=args.cuda)
    test_acc = models.test(netClassifier, test_loader, cuda=args.cuda)
    
    print(f'Accuracy of the model on clean trainset and testset is {trainset_acc}% and {test_acc}%')

    # initialize patch
    patch = patches.init_patch(args)
    
    # TODO: apply statusbar
    # train patch
    patches.train_patch(args, train_loader, test_loader, patch, netClassifier)
    
    