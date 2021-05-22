import matplotlib.pyplot as plt

import pdb                          # The module pdb defines an interactive source code debugger for Python programs
import tensorflow as tf

import math
from matplotlib import pylab as P
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict

import os
import os.path as osp
import numpy as np
import pickle
from io import StringIO ## for Python 3
import PIL.Image
import scipy
import time
import glob
import random

import tensorflow.keras as keras
from tensorflow.keras import applications
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import numpy as np

# pip install git+https://github.com/nottombrown/imagenet_stubs --upgrade
import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import label_to_name, name_to_label


TEST = True


def initialize():
    from os import makedirs
    from wget import download
    
    # Create a directory to store results
    makedirs('cleverhans/results', exist_ok=True)
    
    if not os.path.isfile('cleverhans/toaster.png'):
        imageurl = 'https://user-images.githubusercontent.com/306655/35698271-658aba28-0741-11e8-898b-5a3134634e9e.png'
        download(imageurl, 'cleverhans/toaster.png')

    
if __name__ == '__main__':
    initialize()
    
    # hyperparameters
    TARGET_LABEL = name_to_label('toaster') # Try "banana", "Pembroke, Pembroke Welsh corgi"
    PATCH_SHAPE = (299, 299, 3)
    BATCH_SIZE = 16

    # Ensemble of models
    NAME_TO_MODEL = {
        'xception': applications.xception.Xception,
        'vgg16': applications.vgg16.VGG16,
        'vgg19': applications.vgg19.VGG19,
        'resnet50': applications.resnet50.ResNet50,
        'inceptionv3': applications.inception_v3.InceptionV3,
    }

    MODEL_NAMES = ['resnet50', 'xception', 'inceptionv3', 'vgg16', 'vgg19']

    # Data augmentation
    # Empirically found that training with a very wide scale range works well
    # as a default
    SCALE_MIN = 0.3
    SCALE_MAX = 1.5
    ROTATE_MAX = np.pi/8 # 22.5 degrees in either direction

    MAX_ROTATION = 22.5

    # Local data dir to write files to
    DATA_DIR = 'cleverhans/results'
    
    # Check your GPU devices
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpus)
    # print(tf.test.gpu_device_name())
    
    
    # Image loading
    
    def _convert(im):
        return ((im + 1) * 127.5).astype(np.uint8)

    def show(im):
        plt.axis('off')
        plt.imshow(_convert(im), interpolation="nearest")
        plt.show(block=True)
    
    def load_image(image_path):
        im = PIL.Image.open(image_path)
        im = im.resize((299, 299), PIL.Image.ANTIALIAS)
        if image_path.endswith('.png'):
            ch = 4
        else:
            ch = 3
        im = np.array(im.getdata()).reshape(im.size[0], im.size[1], ch)[:,:,:3]
        return im / 127.5 - 1


    """An image loader that uses just a few ImageNet-like images. 
    In the actual paper, we used real ImageNet images, but we can't include them 
    here because of licensing issues.
    """
    class StubImageLoader():
        def __init__(self):
            self.images = []
            self.toaster_image = None
        
            for image_path in imagenet_stubs.get_image_paths():
                im = load_image(image_path)

                if image_path.endswith('toaster.jpg'):
                    self.toaster_image = im
                else:
                    self.images.append(im)

        def get_images(self):
            return random.sample(self.images, BATCH_SIZE)

    image_loader = StubImageLoader()

    if TEST:
        for example_image in image_loader.get_images()[:2]:
            print("Example true image:")
            show(example_image)
        
        
    # Patch Transformations
    def _transform_vector(width, x_shift, y_shift, im_scale, rot_in_degrees):
        """
        If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
        then it maps the output point (x, y) to a transformed input point 
        (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
        where k = c0 x + c1 y + 1. 
        The transforms are inverted compared to the transform mapping input points to output points.
        """

        rot = float(rot_in_degrees) / 90. * (math.pi/2)
        
        # Standard rotation matrix
        # (use negative rot because tf.contrib.image.transform will do the inverse)
        rot_matrix = np.array(
            [[math.cos(-rot), -math.sin(-rot)],
            [math.sin(-rot), math.cos(-rot)]]
        )
        
        # Scale it
        # (use inverse scale because tf.contrib.image.transform will do the inverse)
        inv_scale = 1. / im_scale 
        xform_matrix = rot_matrix * inv_scale
        a0, a1 = xform_matrix[0]
        b0, b1 = xform_matrix[1]
        
        # At this point, the image will have been rotated around the top left corner,
        # rather than around the center of the image. 
        #
        # To fix this, we will see where the center of the image got sent by our transform,
        # and then undo that as part of the translation we apply.
        x_origin = float(width) / 2
        y_origin = float(width) / 2
        
        x_origin_shifted, y_origin_shifted = np.matmul(
            xform_matrix,
            np.array([x_origin, y_origin]),
        )

        x_origin_delta = x_origin - x_origin_shifted
        y_origin_delta = y_origin - y_origin_shifted
        
        # Combine our desired shifts with the rotation-induced undesirable shift
        a2 = x_origin_delta - (x_shift/(2*im_scale))
        b2 = y_origin_delta - (y_shift/(2*im_scale))
            
        # Return these values in the order that tf.contrib.image.transform expects
        return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)
    

    def test_random_transform(min_scale=0.5, max_scale=1.0,  max_rotation=22.5):
        """
        Scales the image between min_scale and max_scale
        """
        img_shape = [100,100,3]
        img = np.ones(img_shape)
        
        image_in = tf.placeholder(dtype=tf.float32, shape=img_shape)
        width = img_shape[0]
        
        @tf.function
        def _random_transformation():
            im_scale = np.random.uniform(low=min_scale, high=1.0)
            
            padding_after_scaling = (1-im_scale) * width
            x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            
            
            rot = np.random.uniform(-max_rotation, max_rotation)
            
            return _transform_vector(width, 
                                            x_shift=x_delta,
                                            y_shift=y_delta,
                                            im_scale=im_scale, 
                                            rot_in_degrees=rot)

        random_xform_vector = tf.py_func(_random_transformation, [], tf.float32)
        random_xform_vector.set_shape([8])

        output = tf.contrib.image.transform(image_in, random_xform_vector , "BILINEAR")
        
        xformed_img = _random_transformation(output, feed_dict={
            image_in: img
        })
        
        show(xformed_img)

    if TEST:
        for i in range(2):
            print("Test image with random transform: %s" % (i+1))
            test_random_transform(min_scale=0.25, max_scale=2.0, max_rotation=22.5)
            
            