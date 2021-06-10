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
    netClassifier = models.load_openpose() # args.netClassifier
    if args.showProgress:
        print(f'attack model = {netClassifier.__class__}')
    
    
    # load the dataset
    # TODO: apply statusbar
    if args.showProgress:
        print("prepare dataset from =>", args.data_dir)
    #train_loader, test_loader = datasets.load_data(args)
    train_loader, test_loader = datasets.coco_dataloader(args) # src/datasets.py에서 데이터셋 로드  
    
    if args.showProgress:
        print('dataset is loaded. . . complete')
    
    labels = '/home/jubin/Desktop/TeamProject/Adversarial-Patch-for-OpenPose/data/coco/annotations/person_keypoints_val2017.json'
    images_folder = '/home/jubin/Desktop/TeamProject/Adversarial-Patch-for-OpenPose/data/coco/images/val2017/'

    #trainset_acc = models.test(netClassifier, train_loader, cuda=args.cuda, explain=args.showProgress)
    #test_acc = models.test(netClassifier, test_loader, cuda=args.cuda, explain=args.showProgress)
    trainset_acc = models.coco_evaluate(labels, 'detections.json', images_folder, netClassifier) # 로드한 모델에 대해서 정확성 테스트(패치 전)를 진행
    #test_acc = models.evaluate(labels, 'detections.json', images_folder, netClassifier)
    #if args.showProgress:
        #print(f'Accuracy of the model on clean trainset and testset is {trainset_acc}% and {test_acc}%')

    # initialize patch
    patch = patches.init_patch(args)
    
    # TODO: apply statusbar
    # train patch
    patches.train_patch(args, train_loader, test_loader, patch, netClassifier)
    
    