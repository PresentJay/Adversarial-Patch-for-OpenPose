import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def _convert(im):
    return ((im + 1) * 127.5).astype(np.uint8)


def show(im):
    plt.axis('off')
    plt.imshow(_convert(im), interpolation="bilinear")
    plt.show()
    

def show_patched_image(image, probs_patched_image, probs_original_image, true_label, image_index):
    text1 = 'Model prediction (patched image): ' \
        + np.array2string(probs_patched_image, separator = ', ')
    text2 = 'Model prediction (original image): ' \
        + np.array2string(probs_original_image, separator = ', ')
    text3 = 'True label: %d' %true_label
    text4 = 'Image index: %d' % image_index
    text = text1 + '\n' + text2 + '\n' + text3 + '\n' + text4 
    
    plt.axis('off')
    plt.imshow(_convert(image), interpolation="bilinear")
    plt.text(100, -5, text,
        horizontalalignment='center',
        verticalalignment='bottom')
    plt.show()
    
    
def imshow(imgs, label=""):
    imgs = vutils.make_grid(imgs, normalize=True)
    fig, ax = plt.subplots(1, squeeze=False, frameon=False, dpi=300)
    ax[0][0].imshow(np.transpose(imgs.cpu().detach().numpy(), (1, 2, 0)), interpolation="bilinear")  # type: ignore
    ax[0][0].axis("off")  # type: ignore
    ax[0][0].set_title(label)  # type: ignore
    plt.show(block=True)  # type: ignore


def show_tensor(images, title="", text="", block=False):
    images = vutils.make_grid(images, normalize=True)
    
    fig, ax = plt.subplots(1, squeeze=False, frameon=False, dpi=300)
    ax[0][0].imshow(np.transpose(images.cpu().detach().numpy(), (1,2,0)), interpolation='bilinear')
    ax[0][0].axis('off')
    ax[0][0].set_title(title)
    plt.text(122, -20, text,
             horizontalalignment='center',
             verticalalignment='bottom')
    plt.show(block=block)
    
    
def show_numpy(images, title="", text="", block=False):
    if len(images) == 1:
        plt.imshow(images, interpolation="bilinear")
    else:
        # set batch case
        pass
    
    plt.axis('off')
    plt.show(block=block)


def show_batch_data(images, labels, title="", block=False):
    batch = vutils.make_grid(images, nrow=2, padding=20, normalize=True).permute(1,2,0)
    plt.imshow(batch)
    plt.axis('off')
    plt.show(block=block)


def test_random_transform():
    image_shape = [3, 100, 100]
    
    image = np.zeros(image_shape)
    show_numpy(images=image, block=True)
    

# def reducing_rectangle(image_shape, reduce_rate):
#     image_size = (image_size**2) * reduce_rate
#     width = height = int(image_size ** 0.5)
#     return (3, width, height)