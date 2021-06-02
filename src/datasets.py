import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os

import numpy as np

# Load the datasets
def load_data(args):
    
    mean, std = np.array(args.mean, dtype=np.float), np.array(args.std, dtype=np.float)
    
    # Setup the transformation
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(args.image_size, scale=(0.9, 1.0), ratio=(1., 1.)),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(round(args.image_size*1.050)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        # transforms.Resize((args.image_size, args.image_size)),
        # transforms.RandomHorizontalFlip(),              
        transforms.Resize(round(args.image_size*1.050)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # TODO: set seed (in configs.py)
    index = np.arange(args.total_num)
    np.random.shuffle(index)
    train_index = index[:args.train_size]
    test_index = index[args.train_size:(args.train_size + args.test_size)]
    
    train_dataset = ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=train_transforms)
    test_dataset = ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=test_transforms)
    
    # train_dataset = ImageNet(root=args.data_dir, split='train', transform=train_transforms)
    # test_dataset = ImageNet(root=args.data_dir, split='val', transform=test_transforms)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    # pin_memory setting은 GPU환경에서 쓰는 게 좋다! 
    # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    
    return train_loader, test_loader