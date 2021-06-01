import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

# Load the datasets
def load_data(args):
    
    # Setup the transformation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std)
    ])

    # TODO: set seed (in configs.py)
    index = np.arange(args.total_num)
    np.random.shuffle(index)
    train_index = index[:args.train_size]
    test_index = index[args.train_size:(args.train_size + args.test_size)]

    # train_dataset = ImageFolder(root=args.data_dir, transform=train_transforms)
    # val_dataset = ImageFolder(root=args.data_dir, transform=test_transforms)
    
    train_dataset = ImageNet(root=args.data_dir, split='train', transform=train_transforms)
    val_dataset = ImageNet(root=args.data_dir, split='val', transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    # pin_memory setting은 GPU환경에서 쓰는 게 좋다! 
    # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    
    return train_loader, test_loader