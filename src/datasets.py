import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Load the datasets
def dataloader(args):
    
    # Setup the transformation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    index = np.arange(args.total_num)
    np.random.shuffle(index)
    train_index = index[:args.train_size]
    test_index = index[args.train_size: (args.train_size + args.test_size)]

    train_dataset = ImageFolder(root=args.data_dir, transform=train_transforms)
    test_dataset = ImageFolder(root=args.data_dir, transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    return train_loader, test_loader