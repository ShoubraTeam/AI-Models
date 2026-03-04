
# ------------------------------------------------------------------
# This file contains the essential configuration
# ------------------------------------------------------------------

import torch
import torchvision.transforms as T

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




val_transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(
        mean = IMAGENET_MEANs,
        std = IMAGENET_STDs,
    )
])


# val_transform2 = T.Compose([
#     T.Resize(IMAGE_SIZE),
#     T.RandomHorizontalFlip(p = 1),
#     T.ToTensor(),
#     T.Normalize(
#         mean = IMAGENET_MEANs,
#         std = IMAGENET_STDs,
#     )
# ])

# val_transform3 = T.Compose([
#     T.Resize(IMAGE_SIZE),
#     T.ColorJitter(brightness = 0.2, contrast = 0.2),
#     T.ToTensor(),
#     T.Normalize(
#         mean = IMAGENET_MEANs,
#         std = IMAGENET_STDs,
#     )
# ])

# normalize = T.Normalize(mean = IMAGENET_MEANs, std = IMAGENET_STDs)
# val_transform4 = T.Compose([
#     T.Resize((128, 128)), 
#     T.FiveCrop(112),     
#     T.Lambda(lambda crops: torch.stack([normalize(T.ToTensor()(crop)) for crop in crops]))
# ])

# -----------------------------------------------------------------------------
# --- Training CFG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
