RESNET50 = 'resnet50'
INCEPTION_V3 = 'inception_v3' 

def getModels_fromTV(name):
    from torchvision import models
    assert callable(models.__dict__[name]), 'undefined modelname in TorchVision'
    model = models.__dict__[name](pretrained=True).cuda()
    model.eval()
    return model