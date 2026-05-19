# ------------------------------------------------------------------
# Utilities Functions
# ------------------------------------------------------------------

import torch
import torch.nn as nn
from src.utils.models import FaceRecognizerArcFace
import src.utils.confing as CFG
from retinaface.pre_trained_models import get_model
import cv2
import json
import numpy as np





def print_title(title: str, n_sep = 100):
    title = f" {title} "
    title = title.center(n_sep, "=")
    print(title)


def load_arcface_model(path: str):
    # init model
    model = FaceRecognizerArcFace(
        num_classes = CFG.ARCFACE_CFG["N_CLASSES"],
        embedding_dim = CFG.ARCFACE_CFG["EMBEDDING_DIM"],
        margin = CFG.ARCFACE_CFG["MARGIN"]
    )


    # load weights
    loaded = torch.load(path, weights_only = False, map_location = CFG.DEVICE)
    model.load_state_dict(loaded['model_state_dict'])

    model.eval()

    return model


def load_detector():
    backbone_model = "resnet50_2020-07-20"
    retina_face_detector = get_model(
        model_name = backbone_model,
        max_size = CFG.DETECTOR_MAX_SIZE,
        device = CFG.DEVICE
    )   

    retina_face_detector.eval()

    return retina_face_detector


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

def bytes_to_numpy(img_bytes):
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
    return img

def load_obj(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj

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