
# ------------------------------------------------------------
# This file implements utility functions
# - Loading & Displaying Images
# - Connecting Torch & NumPy 
# - Cropping Faces
# - Loading & Saving OBJs
# ------------------------------------------------------------


import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import json
import os
import scripts.config as config

# ---------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- Images Manipulation -------------------------------------------------
def load_image_cv(image_path: str):
    """
    This function takes an img_path and loads the image using opencv

    Args:
        image_path (str)

    Returns:
        img (ndarry)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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
# ----------------------------------------------------- Torch & Numpy -------------------------------------------------------------
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

def tensor_to_ndarry(tensor):
    """ Turns a torch.Tensor (C, H, W) into a numpy array (H, W, C) """
    return tensor.permute(1, 2, 0).cpu().numpy()

def ndarry_to_tensor(ndarry):
    """ Turns a numpy array (H, W, C) into torch.Tensor(C, H, W) """
    return torch.tensor(ndarry, dtype = torch.float32).permute(2, 0, 1)

# ---------------------------------------------------------------------------------------------------------------------------------------
# cropping

def crop_face(face_detector, image):
    """ 
    Detecting the face and cropping it 

    Args:
        retina_detector: the detector
        image: a ndarry image

    Returned:
        cropped: the cropped_face
    """
    detected = face_detector.predict_jsons(image)
    if len(detected) != 1:
        print(f"-- Detector detected {len(detected)} faces")
        return None

    bbox = detected[0]['bbox']
    # bbox = expand_bbox(bbox, image.shape[0], image.shape[1])
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]
    return cropped

def expand_bbox(bbox, img_height, img_width, margin_factor = 0.2):
    """
    Expands the bbox by a percentage of its width & height; to ensure the the detected face covers the whole character

    Args:
        bbox: The original bounding box detected 
        img_height, img_width : The img size
        margin_factor: Expansion percentage

    Returns:
        new_bbox: the new bbox after expansion
    """

    # bbox dimensions
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # padding percentage
    padding_x = int(width * margin_factor)
    padding_y = int(height * margin_factor)

    # expand the bbox
    new_x1 = x1 - padding_x  # to left
    new_y1 = y1 - padding_y
    new_x2 = x2 + padding_x  # to left
    new_y2 = y2 + padding_y

    # ensure the new coordinates are within the image's boundaries 
    new_x1 = max(0, new_x1) # if negative => assign to zero
    new_y1 = max(0, new_y1)
    new_x2 = min(new_x2, img_width) # if > img_width => assign to img_width
    new_y2 = min(new_y2, img_height)

    new_bbox = [
        int(new_x1),
        int(new_y1),
        int(new_x2),
        int(new_y2),
    ]

    return new_bbox
    
# ---------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Objects Loading & Saving -----------------------------------------------
def load_obj(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj

def save_obj(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

# ---------------------------------------------------------------------------------------------------------------------------------------
def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    return obj
# ------------------------------------------------------------------------------------------------------------------------