import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


if os.path.exists("/kaggle/input"):
    ROOT_DATA_DIR = "/kaggle/input/professional-casual-dataset" 
    
else:
    ROOT_DATA_DIR = "data"

if ROOT_DATA_DIR == "data":
    sub_dirs = ["train", "test", "processed"]
    for sub in sub_dirs:
        path = os.path.join(ROOT_DATA_DIR, sub)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

TRAIN_DIR = os.path.join(ROOT_DATA_DIR, "train")
TEST_DIR = os.path.join(ROOT_DATA_DIR, "test")

VAL_SPLIT = 0.2

PATIENCE = 10       
LR_FACTOR = 0.1      
LR_PATIENCE = 3      
MIN_LR = 1e-6

BASELINE_CONFIG = {
    "name": "baseline_cnn",
    "image_size": 64,
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.001,
    "use_pretrained": False,
    "use_scheduler": True,
    "optimizer": "Adam",
    "loss": "BCEWithLogitsLoss"
}

VGG_CONFIG = {
    "name": "vgg16_transfer",
    "image_size": 224,
    "batch_size": 16,
    "epochs": 10,
    "lr": 1e-4,
    "use_pretrained": True,
    "use_scheduler": False,
    "optimizer": "SGD", 
    "loss": "BCEWithLogitsLoss"
}

CURRENT_CONFIG = BASELINE_CONFIG