# -----------------------------------------------------------------
# This file loads the different model arch (including the detector)
# ------------------------------------------------------------------

import torch
from retinaface.pre_trained_models import get_model
from pathlib import Path
from scripts.arch import IResNetSiameseNetwork, FaceRecognizerArcFace
import scripts.config as config

# configurations
BASE_DIR = Path(__file__).resolve().parent.parent

# resnet siamese
RESNET_SIAMESE_ENCODING_DIMENSIONS = 512

# arcface 
ARCFACE_NUM_CLASSES = 479


# ---------------------------------------------------------
def load_model_state_dict(path: str):
    cb = torch.load(path, map_location=config.DEVICE, weights_only=False)
    model_state_dict = cb['model_state_dict']
    return model_state_dict
# ---------------------------------------------------------
def load_siamese_resnet(model_path: str):
    model_path = BASE_DIR / model_path
    model = IResNetSiameseNetwork(encoding_dimensions = RESNET_SIAMESE_ENCODING_DIMENSIONS)
    model.load_state_dict(load_model_state_dict(model_path))
    model = model.to(config.DEVICE).eval()
    return model
# ---------------------------------------------------------
def load_arcface(model_path: str):
    model_path = BASE_DIR / model_path
    model = FaceRecognizerArcFace(num_classes = ARCFACE_NUM_CLASSES)

    # load weigths
    model.load_state_dict(load_model_state_dict(model_path))
    model = model.to(config.DEVICE).eval()

    return model
# ----------------------------------------------------------
def load_retina_detector():
    detector_backbone = "resnet50_2020-07-20"
    retina_face_detector = get_model(
        model_name = detector_backbone,
        max_size = 1024,
        device = config.DEVICE
    )
    retina_face_detector.eval()

    return retina_face_detector
# ---------------------------------------------------------

    
