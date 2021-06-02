import numpy as np
import matplotlib.pyplot as plt

from src import configs, models
from utils import images
from torch.autograd import Variable
import torch

# Initialize the patch
def init_patch(args):
    if args.patch_type == 'rectangle':
        mask_length = int((args.noise_percentage * args.image_size * args.image_size)**0.5)
        patch = np.random.rand(3, mask_length, mask_length)
        return patch


# Generate the mask and apply the patch
def generate_mask(patch, args):
    applied_patch = np.zeros((3, args.image_size, args.image_size))
    
    if args.mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        
        # patch location
        x_location = np.random.randint(low=0, high=args.image_size-patch.shape[1])
        y_location = np.random.randint(low=0, high=args.image_size-patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location


# Test the patch on dataset
def test_patch(args, patch, test_loader, model):
    test_total, test_actual_total, test_success = 0, 0, 0
    print('lets test this patch . . . !')
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] != label and predicted[0].data.cpu().numpy() != args.target:
            print(f'{test_total} : {predicted[0]}vs{label} ~ success')
            
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = generate_mask(patch, args)
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            
            test_success += models.test_image(model, perturbated_image, args.target)
            
        elif predicted[0] == label and predicted[0].data.cpu().numpy() == args.target:
            print(f'exception ==> {test_total} : {label} == {predicted[0]}')
            
    return test_success / test_actual_total


# Patch attack via optimization
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(image, applied_patch, mask, model, args):
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < args.probability_threshold and count < args.max_iteration:
        count += 1
        
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][args.target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = args.lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=0, max=1)
        
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=0, max=1)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][args.target]
        
        if count%100 == 0 :
            print(f'attack {count} : {target_probability}')
    
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch


def train_patch(args, train_loader, test_loader, patch, model):
    best_patch_epoch, best_patch_success_rate = 0.0, 0.0
    
    directoryName = configs.get_current_time()
    configs.init_directories(directoryName)
    
    # TODO: apply statusbar
    # Generate the patch per a epoch
    for epoch in range(args.epochs):
        print(f'{epoch} epoch : patch start . . .')
        train_total, train_actual_total, train_success = 0, 0, 0
        for (image, label) in train_loader:
            train_total += label.shape[0] # 1
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            
            if args.cuda:
                image = image.cuda()
                label = label.cuda()
                
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0] != label and predicted[0].data.cpu().numpy() != args.target:
                print('catched - - - lets make adversarial example ! !')
                print(f'{predicted[0].data.cpu().numpy()} : {args.target}')
                train_actual_total += 1
                
                
                applied_patch, mask, x_location, y_location = generate_mask(patch, args)
                perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, model, args)
                
                perturbated_image = torch.from_numpy(perturbated_image).cuda()
                print('perturbate done. . .')
                images.imshow(perturbated_image)
                
                train_success += models.test_image(model, perturbated_image, args.target)
                patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
        
        
        # - - - show results and save images
        # TODO: move it to utils or so.
        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * args.std + args.mean, 0, 1))
        plt.axis('off')
        plt.savefig(f"results/{directoryName}/candidate/{epoch}.png")
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
        test_success_rate = test_patch(args, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * args.std + args.mean, 0, 1))
            plt.axis('off')
            plt.savefig(f"results/{directoryName}/best/patch.png")

    print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))