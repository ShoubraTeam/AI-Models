
# ------------------------------------------------------------
# This file implements utility functions
# - Loading & Displaying Images
# - Connecting Torch & NumPy 
# - Denormalization of images
# - Loading & Saving OBJs
# ------------------------------------------------------------


import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import json
import os
import face_recognition_config as config


# ---------------------------------------------------------------------------------------------------------------------------------------
def load_image_cv(image_path: str):
    """
    This function takes an img_path and loads the image using opencv
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
# ---------------------------------------------------------------------------------------------------------------------------------------
def show_images(
    images: list,
    titles: list,
    axes = None,
    resize = False
):
    """
    This function displays an img (or images) 

    Args:
        images: a python list contains the images (or a single image)
        titles: a python list contains the string titles
        axes  : Matplotlib axes to plot the images on
    """
    # multiple images given
    if axes is not None:
        assert len(images) == len(list(axes.flat)), "Numbers of given images and axes do not match"
        assert len(titles) == len(list(axes.flat)), "Numbers of given titles and axes do not match"

        for ax, img, tit in zip(axes.flat, images, titles):
            if resize:
                resized = cv2.resize(img, dsize = config.DISPLAY_SIZE)
                ax.imshow(resized)

            else:
                ax.imshow(img)
            ax.set_title(tit)
            ax.axis("off")

        plt.tight_layout(h_pad = 0)
        plt.show()

    # one image given
    else:
        img = images[0]
        tit = titles[0]
        plt.imshow(img)
        plt.title(tit)
        plt.axis("off")
        plt.show()
# ---------------------------------------------------------------------------------------------------------------------------------------

def show_images_with_bboxes(
    images: list,
    titles: list,
    bboxes: list,
    axes = None,
    resize = False,
    color = config.BLUE
):
    """
    This function displays an img (or images) surrounded with their bounding boxes

    Args:
        images: a python list contains the images (or a single image)
        titles: a python list contains the string titles
        bboxes: a python list contains the bounding box for each image.
                    Each bounding box is a list contains
                        x1: left
                        y1: top
                        x2: right
                        y2: bottom
        axes  : matplotlib axes to plot the images on
        color : color used to draw the bounding boxes
    """

    # multiple images given
    if axes is not None:
        assert len(images) == len(list(axes.flat)), "Numbers of given images and axes do not match"
        assert len(titles) == len(list(axes.flat)), "Numbers of given titles and axes do not match"
        assert len(bboxes) == len(list(axes.flat)), "Number of given bboxes does not match"
    
    _images = []
    for idx, img in enumerate(images):
        _img = img.copy()
        bbox = bboxes[idx]
        x1, y1, x2, y2 = bbox

        # draw bbox
        cv2.rectangle(
            img = _img,
            pt1 = (x1, y1),  # top left
            pt2 = (x2, y2),  # bottom right
            color = color,
            thickness = 5
        )

        _images.append(_img)

    # plot imageslist(axes.flat())
    show_images(
        images = _images, 
        titles = titles, 
        axes = axes,
        resize = resize
    )
# ---------------------------------------------------------------------------------------------------------------------------------------
def denormalize_img_tensor(tensor):
    """
    Reverses the normalization operation on a single image tensor using the standard IMAGE_NET means & stds used in normalization.

    Args:
        tensor (torch.Tensor): shape (C, H, W)

    Returns:
        denormalized_tensor (torch.Tensor): shape (C, H, W)
    """

    means_tensor = torch.as_tensor(config.IMAGENET_MEANs, dtype = tensor.dtype, device = tensor.device).view(-1, 1, 1) # (c, 1, 1)
    stds_tensor = torch.as_tensor(config.IMAGENET_STDs, dtype = tensor.dtype, device = tensor.device).view(-1, 1, 1)

    # Normalization process: x_norm = (x - mean) / std
    # Denormalization: x = x_norm * std + mean

    denormalized_tensor = tensor * stds_tensor + means_tensor
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1) # [0,1] range

    return denormalized_tensor
# ---------------------------------------------------------------------------------------------------------------------------------------
def tensor_to_ndarry(tensor):
    """ Turns a torch.Tensor (C, H, W) into a numpy array (H, W, C) """
    return tensor.permute(1, 2, 0).numpy()

def ndarry_to_tenor(ndarry):
    """ Turns a numpy array (H, W, C) into torch.Tensor(C, H, W) """
    return torch.tensor(ndarry, dtype = torch.float32).permute(2, 0, 1)
# ---------------------------------------------------------------------------------------------------------------------------------------

def load_obj(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj

def save_obj(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

# ---------------------------------------------------------------------------------------------------------------------------------------