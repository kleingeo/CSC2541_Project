#Saliency map imports
from vis.visualization import visualize_saliency  # sudo pip install keras-vis
import nibabel as nib
import numpy as np
import matplotlib as plt


def __ProcessImage(img):

    if len(img.shape) > 2:
        img = img[:, :, 0]

    img = (img - img.min()) / (img.max() - img.min()) * 255

    return img


def __GetImage(path, sample_name, slice):

    img_full = nib.load(path + '/' + sample_name)
    seg_img = img_full.get_fdata()[:, :, int(slice)]

    return img_full


def __GetImageData(path, seg_slice):

    #seg_img = __GetImage(path, "", seg_slice)

    image_name_list = ("t2_name", "t1_name", "flair_name", "t1ce_name")

    images = []

    for name in image_name_list:
        img = __GetImage(path, name, seg_slice)
        img = __ProcessImage(img)
        images.append(img)

    return images


def __ConcatData(imgs):

    x_data = imgs[0]

    if imgs[1] is not None:
        x_data = np.stack([x_data, imgs[1]], axis=-1)

    if imgs[2] is not None:
        if len(x_data.shape) > 2:
            flair_img = np.expand_dims(imgs[2], axis=-1)
            x_data = np.concatenate([x_data, flair_img], axis=-1)
        else:
            x_data = np.stack([x_data, imgs[2]], axis=-1)

    if imgs[3] is not None:
        if len(x_data.shape) > 2:
            t1ce_img = np.expand_dims(imgs[3], axis=-1)
            x_data = np.concatenate([x_data, t1ce_img], axis=-1)

    if (imgs[1] is None) and (imgs[2] is None):
        x_data = np.expand_dims(x_data, axis=-1)

    #y_data = np.expand_dims(seg_img, axis=-1)

    return x_data


def __ShowImg(img, grads):

    plt.imshow(img)
    plt.imshow(grads, alpha=.6)
    plt.axis('off')
    plt.imshow(grads)


def SaliencyMap(model):

    path = ""
    selected_slice = 0

    images = __GetImageData(path, seg_slice=selected_slice)
    input_image = __ConcatData(images)

    true_label = None  # 0  # label?
    layer_idx = -1  # last layer of the model
    grads = visualize_saliency(model, layer_idx, filter_indices=true_label, seed_input=input_image)

    __ShowImg(input_image, grads)