import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.coco import CocoTrainDataset, CocoValDataset
from src.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip

COCO_MEAN = (0.4404, 0.4440, 0.4327)
COCO_STD = (0.2458, 0.2410, 0.2468)

ROOT = '/home/jubin/Desktop/imagenet-mini'
jsonfile = './data/mpii/annotation/mpii_annotations.json'


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


def coco_dataloader(args):
    prepared_train_labels = '/home/jubin/Desktop/TeamProject/Adversarial-Patch-for-OpenPose/data/coco/annotations/prepared_train_annotation.pkl'
    train_images_folder = '/home/jubin/Desktop/TeamProject/Adversarial-Patch-for-OpenPose/data/coco/images/train2017/'
    labels = '/home/jubin/Desktop/TeamProject/Adversarial-Patch-for-OpenPose/data/coco/annotations/person_keypoints_val2017.json'
    images_folder = '/home/jubin/Desktop/TeamProject/Adversarial-Patch-for-OpenPose/data/coco/image/val2017/'
    stride = 8
    sigma = 7
    path_thickness = 1

    train_dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
                               stride, sigma, path_thickness,
                               transform=transforms.Compose([
                                   ConvertKeypoints(),
                                   Scale(),
                                   Rotate(pad=(128, 128, 128)),
                                   CropPad(pad=(128, 128, 128)),
    
                                   Flip()]))

    val_dataset = CocoValDataset(labels, images_folder)
    
    print(len(train_dataset))
    print(len(val_dataset))
    for image in train_dataset:
        print(image)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    #for batch in val_dataset:
    #    print(batch)
    #    break

    return train_loader, val_loader

