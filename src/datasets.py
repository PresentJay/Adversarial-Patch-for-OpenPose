import os
import numpy as np

import torch
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


class DataSet():
    def __init__(self, source, name, shape, trainfull, trainsize, testfull, testsize, explain):
        self.name = name
        self.shape = shape
        self.explain = explain

        if self.shape[1] == 299:    # when input image shape is 299x299 something
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif self.shape[1] == 244:  # when input image shape is 244x244 something
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        # expandable area <---

        # ---/>

        # TODO: figure out the space to EOT variables.
        # self.transformed = kwargs.pop('transformed')

        self.train_index = self.Shuffle(fullsize=trainfull, size=trainsize)
        self.test_index = self.Shuffle(fullsize=testfull, size=testsize)

        self.LoadByFolder(source)
        # LoadByImageNet(source)

        if self.explain:
            print(f'dataset {self.name} is loaded from [{source}].')
            print(
                f'train data size is {trainsize}, test data size is {testsize}.')

    def Shuffle(self, fullsize, size):
        index = np.arange(fullsize)
        np.random.shuffle(index)
        return index[:size]

    def Prepare(self):
        # if self.transformed:

        return transforms.Compose(
            [
                # plain prepare
                transforms.Resize((self.shape[1], self.shape[2])),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )

        """ else:
            # prepare Expectation Over Transformation
            return transforms.Compose(
                [
                    # To prepare EOT variables area <---
                    
                    # ---/>
                    
                    # plain prepare
                    transforms.Resize((self.shape[1], self.shape[2])),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]
            ) """

    def LoadByFolder(self, source):
        self.trainset = ImageFolder(root=os.path.join(
            source, 'train'), transform=self.Prepare())
        self.testset = ImageFolder(root=os.path.join(
            source, 'val'), transform=self.Prepare())

    def LoadByImageNet(self, source):
        self.trainset = ImageNet(
            root=source, split="train", transform=self.Prepare())
        self.testset = ImageNet(root=source, split="val",
                                transform=self.Prepare())

    def SetDataLoader(self, batch_size, num_workers, pin_memory=True, shuffle=False):
        # pin_memory setting is good for GPU environments!
        # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

        self.train_loader = DataLoader(
            dataset=self.trainset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(self.train_index),
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )

        self.test_loader = DataLoader(
            dataset=self.testset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(self.test_index),
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )

        if self.explain:
            print('. . . dataloaders are ready.')

    def GetTrainData(self):
        return self.train_loader

    def GetTestData(self):
        return self.test_loader


# Load the datasets
def load_data(args):
    mean, std = np.array(args.mean, dtype=np.float), np.array(
        args.std, dtype=np.float)

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

    train_dataset = ImageFolder(root=os.path.join(
        args.data_dir, 'train'), transform=train_transforms)
    test_dataset = ImageFolder(root=os.path.join(
        args.data_dir, 'val'), transform=test_transforms)

    # train_dataset = ImageNet(root=args.data_dir, split='train', transform=train_transforms)
    # test_dataset = ImageNet(root=args.data_dir, split='val', transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(
        train_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(
        test_index), num_workers=args.num_workers, pin_memory=True, shuffle=False)

    # pin_memory setting은 GPU환경에서 쓰는 게 좋다!
    # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

    return train_loader, test_loader

# --------------- COCO ----------------


def coco_dataloader(args):
    prepared_train_labels = 'data/annotations/prepared_train_annotation.pkl'
    train_images_folder = 'D:/datasets/coco2017/images/train2017/'
    labels = 'data/annotations/person_keypoints_val2017.json'
    images_folder = 'D:/datasets/coco2017/images/val2017/'

    stride = 8
    sigma = 7
    path_thickness = 1

    train_dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
                                     stride, sigma, path_thickness,
                                     transform=transforms.Compose([
                                            transforms.Resize((244, 244)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(COCO_MEAN, COCO_STD)
                                         ]))

    val_dataset = CocoValDataset(labels, images_folder)

    # print(len(train_dataset))
    # print(len(val_dataset))
    # for image in train_dataset:
    #     print(image)
    #     input()
    #     break

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    return train_loader, val_loader
