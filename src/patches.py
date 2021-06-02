import numpy as np
import matplotlib.pyplot as plt
import math

from src import configs, models
from utils import images, times
from torch.autograd import Variable
import torch
import torch.nn.functional as F


# Initialize the patch
def init_patch(args):
    if args.patch_shape == 'rectangle':
        mask_length = int((args.noise_percentage * args.image_size * args.image_size)**0.5)
        patch = np.random.rand(3, mask_length, mask_length)
        return patch
    

def init_patch_rectangle(image_size, patch_size):
    # get mask
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random.rand(1,3,noise_dim,noise_dim)
    return patch, patch.shape


def rectangle_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
    
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(4)
        for j in range(patch[i].shape[0]):
            patch[i][j] = np.rot90(patch[i][j], rot)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
       
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask
    
    
def init_patch_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi))
    patch = np.zeros((1, 3, radius*2, radius*2))    
    for i in range(3):
        a = np.zeros((radius*2, radius*2))    
        cx, cy = radius, radius # The center of circle 
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))
        
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


# Generate the mask and apply the patch
def generate_mask(patch, args):
    applied_patch = np.zeros((3, args.image_size, args.image_size))
    
    if args.patch_shape == 'rectangle':
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
    if args.showProgress:
        print('lets test this patch . . . !')
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] != label and predicted[0].data.cpu().numpy() != args.target:
            if args.showProgress:
                print(f'{test_total} : {predicted[0]} vs {label.data[0]}')
            
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = generate_mask(patch, args)
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            
            test_success += models.test_image(model, perturbated_image, args.target)
            
    return test_success / test_actual_total


# Patch attack via optimization
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(image, patch, mask, model, args):
    patch = torch.from_numpy(patch).type(torch.FloatTensor)
    mask = torch.from_numpy(mask).type(torch.FloatTensor)
    image = image.type(torch.FloatTensor)
    print('patch attack start . . .')
    
    if args.cuda:
        patch = patch.cuda()
        mask = mask.cuda()
        image = image.cuda()
    
    # compute the probability to target of the original image
    output = F.softmax(model(image), dim=1)
    target_probability = output.data[0][args.target]
    print('start pribability:', target_probability)
        
    image_with_patch = torch.mul(mask, patch) + torch.mul((1 - mask), image)
    
    for count in range(args.max_iteration):
        # Optimize the patch
        image_with_patch = Variable(image_with_patch.data, requires_grad=True)
        
        if args.cuda:
            image_with_patch = image_with_patch.cuda()
        
        adversarial_output = F.log_softmax(model(image_with_patch), dim=1)
        
        loss = -adversarial_output[0][args.target]
        loss.backward()
        
        adversarial_gradient = image_with_patch.grad.clone()
        image_with_patch.grad.data.zero_()
        
        
        patch = patch - adversarial_gradient
        
        image_with_patch = torch.mul(mask, patch) + torch.mul((1 - mask), image)
        image_with_patch = torch.clamp(image_with_patch, min=0, max=1)
        
        output = F.softmax(model(image_with_patch), dim=1)
        target_probability = output.data[0][args.target]
        
        images.imshow(image_with_patch, label= f'{count+1} attacks image : {target_probability*100}% for {args.target}')
            
        if target_probability >= args.probability_threshold:
            break
    
    image_with_patch = image_with_patch.cpu().numpy()
    patch = patch.cpu().numpy()
    
    if args.showProgress:
        print(f'{count} of attack is done. . . success rate : {target_probability * 100}% ')
    
    return image_with_patch, patch


def train_patch(args, train_loader, test_loader, patch, model):
    best_patch_epoch, best_patch_success_rate = 0, 0
    
    directoryName = times.get_current_time()
    configs.init_directories(directoryName)
    
    # TODO: apply statusbar
    # Generate the patch per a epoch
    for epoch in range(args.epochs):
        if args.showProgress:
            print(f'{epoch} epoch : patch start . . .')
            
        train_total, train_actual_total, train_success = 0, 0, 0
        for (image, label) in train_loader:
            train_total += label.shape[0] # 1
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            
            if args.cuda:
                image = image.cuda()
                label = label.cuda()
                
            output = model(image)
            prediction = output.data.max(1)[1][0]
            if prediction != args.target:
                if args.showProgress:
                    print(f'{epoch}-{train_total} : catched - - - lets make adversarial example ! !')
                    print(f'target {args.target} : prediction {output.data.max(1)[1][0]}')
                train_actual_total += 1
                
                applied_patch, mask, x_location, y_location = generate_mask(patch, args)
                perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, model, args)
                
                perturbated_image = torch.from_numpy(perturbated_image).cuda()
                if args.showProgress:
                    print('perturbate done. . .')
                    # images.imshow(perturbated_image)
                
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

    
