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


IMAGE_SIZE = (112, 112)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEANs = [0.5, 0.5, 0.5]
STDs = [0.5, 0.5, 0.5]