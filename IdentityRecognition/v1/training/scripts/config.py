# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# ------------------------------------------------------------------
# This file contains the essential configuration
# ------------------------------------------------------------------

import torch
import torchvision.transforms as T
import random



# --- Displaying CFG ---
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
DISPLAY_SIZE = (300, 300)

RED_TEXT_COLOR = '\033[91m'
GREEN_TEXT_COLOR = '\033[92m'
RESET_TEXT = '\033[0m'
BULLET = "•"

# -----------------------------------------------------------------------------

IMAGE_SIZE = (112, 112)
IMAGENET_MEANs = [0.485, 0.456, 0.406]
IMAGENET_STDs  = [0.229, 0.224, 0.225]


# transforms
train_transform = T.Compose([
    T.RandomHorizontalFlip(p = 0.5),
    T.RandomRotation(degrees = 15),
    
    T.RandomResizedCrop(
        size = IMAGE_SIZE, 
        scale = (0.7, 1.0), 
        ratio = (0.9, 1.1)
    ),

    T.ColorJitter(
        brightness = 0.4, 
        contrast = 0.4, 
        saturation = 0.3, 
        hue = 0.1
    ),
    
    T.RandomGrayscale(p = 0.3),

    T.RandomApply([
        T.GaussianBlur(kernel_size = (3, 5), sigma = (0.1, 3.0))
    ], p = 0.3), 
    
    T.RandomApply([
        T.RandomAdjustSharpness(sharpness_factor = 5, p = 1.0)
    ], p = 0.3), 
    
    T.ToTensor(),
    
    T.Lambda(lambda x: x + torch.randn_like(x) * 0.08 if random.random() < 0.3 else x),
    
    T.Normalize(
        mean = IMAGENET_MEANs,
        std = IMAGENET_STDs,
    )
])


val_transform = T.Compose([
    T.Resize(size = IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(
        mean = IMAGENET_MEANs,
        std = IMAGENET_STDs,
    )
])

val_transform_vgg = T.Compose([
    T.Resize(size = (224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean = IMAGENET_MEANs,
        std = IMAGENET_STDs,
    )
])


# -----------------------------------------------------------------------------
# --- Training CFG VGG Siamese ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
TRAIN_EPOCH_SIZE = 192000
VAL_EPOCH_SIZE = 64000
INPUT_SHAPE = (3, 224, 224)
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-4

# Trained Parameters
TRAINED_MODEL_HIDDEN_LAYERS = 4
TRAINED_MODEL_HIDDEN_NEURONS = 100
UNFROZEN_BACKBONE = 17
VGG_OP_DIMENSIONS = 25088
LEARNED_THRESHOLD = 1.2329