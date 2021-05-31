import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROOT = 'D:\datasets\ImageNet'

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

    # train_dataset = ImageFolder(root=args.data_dir, transform=train_transforms)
    # test_dataset = ImageFolder(root=args.data_dir, transform=test_transforms)
    
    train_dataset = ImageNet(root=ROOT, split='train', transform=train_transforms)
    val_dataset = ImageNet(root=ROOT, split='val', transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    # pin_memory setting은 GPU환경에서 쓰는 게 좋다! 
    # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    
    return train_loader, test_loader