import torch

# Test the model on clean dataset
def test(model, dataloader):
    model.eval()
    correct, total, loss, count = 0, 0, 0, 1
    with torch.no_grad():
        for (images, labels) in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            
            if count % 1000 == 0:
                print(f'{count} of test image is loaded. . .')
            
            count+=1
            
    return (correct / total) * 100