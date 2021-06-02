import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def _convert(im):
    return ((im + 1) * 127.5).astype(np.uint8)


def show(im):
    plt.axis('off')
    plt.imshow(_convert(im), interpolation="nearest")
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
    plt.imshow(_convert(image), interpolation="nearest")
    plt.text(100, -5, text,
        horizontalalignment='center',
        verticalalignment='bottom')
    plt.show()
    
    
def imshow(imgs, label=""):
    imgs = vutils.make_grid(imgs, normalize=True)
    fig, ax = plt.subplots(1, squeeze=False, frameon=False, dpi=300)
    ax[0][0].imshow(np.transpose(imgs.cpu().detach().numpy(), (1, 2, 0)), interpolation="nearest")  # type: ignore
    ax[0][0].axis("off")  # type: ignore
    ax[0][0].set_title(label)  # type: ignore
    plt.show(block=True)  # type: ignore
