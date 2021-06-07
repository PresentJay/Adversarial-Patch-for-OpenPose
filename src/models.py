import torch
from torchvision import models

from src import coco

def getModels_fromTV(name):
    assert callable(models.__dict__[name]), 'undefined modelname in TorchVision'
    model = models.__dict__[name](pretrained=True).cuda()
    model.eval()
    return model


def listModels_in_TV():
    alldict = set(models.__dict__.keys())
    filterset = set(['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__path__', '__file__', '__cached__', '__builtins__', 'utils', '_utils', 'segmentation', 'detection', 'video', 'quantization'])
    return list(alldict - filterset)


def test(model, dataloader, cuda=False, explain=True):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for index, (images, labels) in enumerate(dataloader):
            if cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            
            # rank 1
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if total % 500 == 0 and explain:
                print(f'{total} of images are tested . . . intermediate score = {correct / total * 100}')
    
    return correct / total * 100

def coco_evaluate(labels, output_name, images_folder, net, multiscale=False, visualize=False):
    net = net.cuda().eval()
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = coco.CocoValDataset(labels, images_folder)
    coco_result = []
    for sample in dataset:
        file_name = sample['file_name']
        img = sample['img']

        avg_heatmaps, avg_pafs = coco.infer(net, img, scales, base_height, stride)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += coco.extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = coco.group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = coco.convert_to_coco_format(pose_entries, all_keypoints)

        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey()
            if key == 27:  # esc
                return

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)


def predict_once(model, image, target=None, explain=True):
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    prediction = predicted[0].data.cpu().numpy()
    
    if target is None:
        return prediction
    elif explain:
        print(f'test image - - - {prediction} : {target}')
        print('success\n' if prediction == target else 'failed\n')
        return 1 if prediction == target else 0
    
    
def load_openpose():
    import torch
    model = torch.load('./data/models/pose_model.pth')
    #model.eval()