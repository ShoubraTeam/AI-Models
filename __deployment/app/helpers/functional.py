# ---------------------------------------------
# Import general utility functions
# ---------------------------------------------

import os
from models.data_config import FEATURE_ALLOWED_FEATURES, FEATURE_IDENITY_RECOGNITION
from helpers.config import get_settings
from typing import Any

import torch
from agents import FaceRecognizerArcFace
from helpers.config import ARCFACE_CFG, RETINA_DETECTOR_CFG


from agents import FaceRecognizerArcFace
from retinaface.pre_trained_models import get_model as get_retina_model

import traceback
import json
from pathlib import Path

import weaviate
import os
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate import WeaviateClient

from langchain_huggingface import HuggingFaceEmbeddings


settings = get_settings()

def validate_feature_id(feature_id: str) -> bool:
    """
    Validate the given feature ID
        
    Args:
        feature_id (str)
    
    Returns:
        is_ok (bool)
    """
    if feature_id not in FEATURE_ALLOWED_FEATURES:
        return False
    
    return True




# -------------------- Printing Utils --------------------------
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"





def format_error(error: Exception) -> dict:
    tb = traceback.extract_tb(error.__traceback__)

    last_frame = tb[-1] if tb else None

    return {
        "error_type"    : type(error).__name__,
        "error_message" : str(error),
        "error_file"    : str(Path(last_frame.filename).resolve()) if last_frame else None,
        "error_line"    : last_frame.lineno if last_frame else None,
        "error_function": last_frame.name if last_frame else None,
    }

def print_subtitle(subtitle: str):
    print()
    subtitle = f" {subtitle} ".center(50, "=")
    print(f"{BLUE}{subtitle}{RESET}")
    print()


def print_success_message(message: str):
    print(f"{GREEN}>> {message} <<{RESET}")



def print_error(error: Exception, message: str):
    sep = 50 * '='
    print(f"{RED}{sep}{RESET}")

    print(f"{RED}>> {message} <<{RESET}: ")
    
    err_json = format_error(error)
    print(json.dumps(err_json, indent = 2))
    
    print(f"{RED}{sep}{RESET}")




def print_title(title: str, n_sep: int =  100, sep: str = "="):
    """Printing a title in a well-formatted manner"""
    title = f" {title} ".center(n_sep, sep)
    print(title)


def print_structured_response(structured_response):
    """Printing the Agent Structured Response"""
    # print_title("Structured Response", 50)
    structured_response = structured_response.model_dump()

    for attb, value in structured_response.items():
        print()
        print(f"{attb.capitalize()}", end = "")

        if isinstance(value, list):
            print(":")
            for item in value:
                if isinstance(item, dict): 
                    for key, val in item.items():
                        print(f"\t{key} => {val}")
                else:
                    print(f"\t{item}")
                
                print()

        else:
            print(f"=> {value}")


def print_dict(dic: dict):
    print()
    for k, v in dic.items():
        print(f"{k} => {v}")
    
def print_semi_dict(semi_dict: Any):
    print()
    items = semi_dict.__dict__.items()
    for k, v in items:
        print(f"{k} => {v}")

    

def print_data(data: Any):
    if isinstance(data, list):
        if isinstance(data[0], dict):
            for dic in data:
                print_dict(dic)
        
        elif hasattr(data[0], "__dict__"):
            for dic in data:
                print_semi_dict(dic)
        
        else:
            print()
            print(data)

    elif isinstance(data, dict):
        print_dict(data)
    
    elif hasattr(data, "__dict__"):
        print_semi_dict(data)
    
    else:
        print()
        print(data)
    


def print_eval_data(eval_data: list[dict]):
    print_subtitle("Extracted Eval Data:")
    
    for idx, sample in enumerate(eval_data, start = 1):
        print(f"Sample #{idx}")

        job_data = sample["job"]
        for k, v in job_data.items():
            print(f">> {k}:\n{v}\n\n")

        proposals = sample["proposals"]
        for proposal_idx, proposal in enumerate(proposals, start = 1):
            print(f"Proposal #{proposal_idx}")

            for k, v in proposal.items():
                print(f">> {k}:\n{v}\n\n")
            
        print("------")


# ------------------- Loading Agents / Clients --------------------------

# identity recognition
def get_identity_recognizer() -> FaceRecognizerArcFace:
    weights_path = os.path.join(
        settings.TRAINED_MODELS_PATH,
        FEATURE_IDENITY_RECOGNITION,
        "arcface_model.pth"
    )

    model = FaceRecognizerArcFace(
        num_classes = ARCFACE_CFG["n_classes"],
        embedding_dim = ARCFACE_CFG["embedding_dim"],
        margin = ARCFACE_CFG["margin"]
    )


    # load weights
    loaded = torch.load(weights_path, map_location = ARCFACE_CFG["device"])
    model.load_state_dict(loaded['model_state_dict'])

    model.eval()

    return model


def get_retina_face_detector():
    backbone_model = "resnet50_2020-07-20"
    retina_face_detector = get_retina_model(
        model_name = backbone_model,
        max_size = RETINA_DETECTOR_CFG["max_size"],
        device = RETINA_DETECTOR_CFG["device"]
    )   

    retina_face_detector.eval()

    return retina_face_detector


# job description enhancement
def get_weaviate_client() -> WeaviateClient:
    """
    Returns:
        client: the Weaviate API required to use the database
    """
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url = settings.WEAVIATE_URL,
        auth_credentials = Auth.api_key(api_key = settings.WEAVIATE_API_KEY),
        additional_config = AdditionalConfig(
            timeout = Timeout(init = 30, query = 60, insert = 120)
        )
    )

    return client

from helpers.config import JOB_DESCRIPTION_RAG_EMBEDDER, JOB_DESCRIPTION_RAG_RERANKER
from sentence_transformers import CrossEncoder
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name    = JOB_DESCRIPTION_RAG_EMBEDDER["model_name"],
        model_kwargs  = JOB_DESCRIPTION_RAG_EMBEDDER["model_kwargs"],
        encode_kwargs = JOB_DESCRIPTION_RAG_EMBEDDER["encode_kwargs"]
    )

def get_raranker() -> CrossEncoder:
    return CrossEncoder(JOB_DESCRIPTION_RAG_RERANKER)


from controllers import WeaviateController
from weaviate.collections import Collection
from models.enums import ErrorsEnum
def get_weaviate_collection(agents, client) -> Collection:
    weaviate_controller = WeaviateController(
        agents = agents,
        client = client
    )

    collection = weaviate_controller.get_collection()
    if collection is None:
        try:
            collection = weaviate_controller.build_collection()
            collection = weaviate_controller.fill_collection()
        except Exception as e:
            print_error(error = e, message = ErrorsEnum.DEBUG_WEAVIATE_BUILD_ERROR.value)
            raise

    return collection