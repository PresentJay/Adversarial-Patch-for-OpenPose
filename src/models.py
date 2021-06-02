from torchvision import models
import torch

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


def test_image(model, image, target=None, explain=True):
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    prediction = predicted[0].data.cpu().numpy()
    
    if target is None:
        return prediction
    elif explain:
        print(f'test image - - - {prediction} : {target}')
        print('success\n' if prediction == target else 'failed\n')
        return 1 if prediction == target else 0