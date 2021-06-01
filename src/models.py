from torchvision import models


def getModels_fromTV(name):
    assert callable(models.__dict__[name]), 'undefined modelname in TorchVision'
    model = models.__dict__[name](pretrained=True).cuda()
    model.eval()
    return model


def listModels_in_TV():
    alldict = set(models.__dict__.keys())
    filterset = set(['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__path__', '__file__', '__cached__', '__builtins__', 'utils', '_utils', 'segmentation', 'detection', 'video', 'quantization'])
    return list(alldict - filterset)
