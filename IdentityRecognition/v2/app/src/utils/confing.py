# ------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------



import torch
import torchvision.transforms as T

ARCFACE_CFG = {
    "N_CLASSES" : 786,
    "EMBEDDING_DIM" : 512,
    "MARGIN" : 0.5
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTOR_MAX_SIZE = 768

IMAGE_SIZE = (112, 112)
MEANs = [0.5, 0.5, 0.5]
STDs = [0.5, 0.5, 0.5]
TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(
        mean = MEANs,
        std = STDs
    )
])