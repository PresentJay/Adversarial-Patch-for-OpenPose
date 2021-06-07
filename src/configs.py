import argparse
import time
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def init_args():
    parser = argparse.ArgumentParser()
    
    # about Environments of this experiment
    parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')
    parser.add_argument('--GPU', type=str, default='0', help="index of used GPU")
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--showProgress', action='store_true', default=True, help='show process logs for your understanding')
    parser.add_argument('--num_workers', type=int, default=4, help="num_workers (to be half of your CPU cores")
    
    # about Dataset
    parser.add_argument('--total_num', type=int, default=200, help="number of dataset images")
    parser.add_argument('--train_size', type=int, default=100, help="number of training images")
    parser.add_argument('--test_size', type=int, default=100, help="number of test images")
    parser.add_argument('--data_dir', type=str, default='D:\datasets\ImageNet', help="dir of the dataset")
    parser.add_argument('--image_size', type=int, default=244, help='the height / width of the input image to network (basically 244, inception_v3 is 299')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    
    # about Model (pretrained)
    parser.add_argument('--netClassifier', default='vgg19', help="The target classifier")
    
    # about Training Adversarial Patch
    parser.add_argument('--noise_percentage', type=float, default=0.1, help="percentage of the patch size compared with the image size")
    parser.add_argument('--max_iteration', type=int, default=1000, help="max number of iterations to find adversarial example")
    parser.add_argument('--patch_size', type=float, default=0.5, help='patch size. E.g. 0.05 ~= 5% of image ')
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
    parser.add_argument('--epochs', type=int, default=5, help="total epoch")

    # about Adversarial Patch Condition
    parser.add_argument('--target', type=int, default=859, help="target label index")
    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch => (circle or rectangle)")
    parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
    
    # about logging
    parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
 
    args = parser.parse_args()
    assert args.train_size + args.test_size <= args.total_num, "train_size + test_size must be same or lower than Total dataset size"
    
    if args.netClassifier.startswith('inception'):
        assert args.image_size==299, "inception model's input is 299"
        
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    
    return args


def init_directories(directoryName):
    try:
        os.makedirs(f'results/{directoryName}/candidate', exist_ok=True)
        os.makedirs(f'results/{directoryName}/best', exist_ok=True)
    
    except OSError:
        pass
    