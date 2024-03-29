import numpy as np
import matplotlib.pyplot as plt
import math

from src import configs, models
from utils import images, times
from torch.autograd import Variable

import torch
import torch.nn.functional as F

class AdversarialPatch():
    def __init__(self, dataset, target, device, _type, explain):
        self.dataset = dataset
        self.device = device
        self.explain = explain
        self._type = _type
        self.target = target
        
        # self.shape = imgUtil.reducing_rectangle(dataset.shape, reduce_rate=reduce_rate)
        
        mean = torch.tensor(self.dataset.mean, dtype=torch.float)
        std = torch.tensor(self.dataset.std, dtype=torch.float)
        val = lambda x: ((x - mean) / std).to(self.device).unsqueeze(1).unsqueeze(1)
        self.val = {'min': val(0), 'max': val(1)}
        
        self.adversarial_image = self.init_Adversarial_Image()

        # self.mask = self.init()
    
    
    
    def init_Adversarial_Image(self, random_init=False):
        if random_init:
            adversarial_image = torch.rand(self.dataset.shape).to(self.device)
        else:
            adversarial_image = torch.zeros(self.dataset.shape).to(self.device)
        
        # rand val[min] to val[max]
        adversarial_image = adversarial_image * (self.val['max'] - self.val['min']) + self.val['min']
        
        if self.explain:
            images.show_tensor(adversarial_image, block=True)
            
        return adversarial_image
    
    
    def clamp_patch_to_valid(self, patch):
        ch_ranges = [
            [-self.dataset.mean[0] / self.dataset.std[0], (1 - self.dataset.mean[0]) / self.dataset.std[0]],
            [-self.dataset.mean[1] / self.dataset.std[1], (1 - self.dataset.mean[1]) / self.dataset.std[1]],
            [-self.dataset.mean[2] / self.dataset.std[2], (1 - self.dataset.mean[2]) / self.dataset.std[2]]
        ]
        with torch.no_grad():
            self.patch[0] = torch.clamp(self.patch[0], ch_ranges[0][0], ch_ranges[0][1])
            self.patch[1] = torch.clamp(self.patch[1], ch_ranges[1][0], ch_ranges[1][1])
            self.patch[2] = torch.clamp(self.patch[2], ch_ranges[2][0], ch_ranges[2][1])

    
    def init_mask(self):        
        width = self.adversarial_image.shape[1]
        height = self.adversarial_image.shape[2]
        
        min_index = np.argmin(self.shape)
        print(min_index)
        
        start = (self.shape[min_index-1] - self.shape[min_index])/2
        print(start)
        
        # make shape to circle
        if _type == 'circle':
            pass
        
        return mask
    
    
    # train patch
    def train(self, model):
        model.eval()
        
        success = total = 0
        
        


# Initialize the patch
def init_patch(args):
    image_size = args.image_size ** 2
    noised_image_size = image_size * args.patch_size
    
    if args.patch_type == 'rectangle':
        mask_length = int(noised_image_size ** 0.5)
        patch = np.random.rand(1, 3, mask_length, mask_length)
    
    elif args.patch_type == 'circle':
        radius = int(math.sqrt(noised_image_size / math.pi))
        patch = np.zeros((1, 3, radius * 2, radius * 2))
        for dim in range(3):
            circle = np.zeros((radius*2, radius*2))
            x, y = np.ogrid[-radius:radius, -radius:radius]
            index = (x**2 + y**2) <= radius**2
            circle[0:radius*2, 0:radius*2][index] = np.random.rand()
            idx = np.flatnonzero((circle == 0).all((1)))
            circle = np.delete(circle, idx, axis=0)
            patch[0][i] = np.delete(circle, idx, axis=1)
    
    return patch



# transpormation the patch by using mask
def generate_mask(patch, patch_shape, image_shape, args):
    mask = np.zeros(image_shape)
    
    if args.patch_type == 'rectangle':
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
        
        adversarial_output = F.log_softmax(model
                                           (image_with_patch), dim=1)
        
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
            
            # print(f'image_shape : {image.cpu().numpy().shape}')  # check the image_shape
                
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
                
                train_success += models.predict_once(model, perturbated_image, args.target)
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

    
