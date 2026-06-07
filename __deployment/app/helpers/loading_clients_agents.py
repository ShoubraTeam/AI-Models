# --------------------------------------------
# Loading:
# - Agents
# - Models
# - Weaviate collection
# - Clients
# --------------------------------------------

import os
import torch

from retinaface.pre_trained_models import get_model as get_retina_model
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from helpers.config import ARCFACE_CFG, RETINA_DETECTOR_CFG
from helpers.config import JOB_DESCRIPTION_RAG_EMBEDDER, JOB_DESCRIPTION_RAG_RERANKER
from helpers.config import get_settings
from helpers.functional import print_error

from models.data_config import FEATURE_IDENITY_RECOGNITION
from models.enums import ErrorsEnum

from controllers import WeaviateController
from agents import FaceRecognizerArcFace

import weaviate
from weaviate.collections import Collection
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate import WeaviateClient


settings = get_settings()


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


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name    = JOB_DESCRIPTION_RAG_EMBEDDER["model_name"],
        model_kwargs  = JOB_DESCRIPTION_RAG_EMBEDDER["model_kwargs"],
        encode_kwargs = JOB_DESCRIPTION_RAG_EMBEDDER["encode_kwargs"]
    )

def get_raranker() -> CrossEncoder:
    return CrossEncoder(JOB_DESCRIPTION_RAG_RERANKER)



def get_weaviate_collection(client) -> Collection:
    weaviate_controller = WeaviateController(
        agents = None,
        client = client
    )

    collection = weaviate_controller.get_collection()
    if collection is None:
        print("\t>> Collection Does not Exist, Wait while building it")
        try:
            collection = weaviate_controller.build_collection()
            collection = weaviate_controller.fill_collection()
        except Exception as e:
            print_error(error = e, message = ErrorsEnum.DEBUG_WEAVIATE_BUILD_ERROR.value)
            raise

    return collection