# --------------------------------------------
# Loading:
# - Agents
# - Models
# - Weaviate collection
# - Clients
# --------------------------------------------

# ---------------------------------------- Imports ---------------------------------- #

# general
import os

from .startup_noise import configure_startup_noise, suppress_model_loader_output
configure_startup_noise()

from typing import Any, TypeAlias
from helpers.functional import print_error
from helpers.settings import get_settings

from transformers import pipeline

# helpers & config
from models.enums import ErrorsEnum
from models.config.system_tasks import FEATURE_IDENITY_RECOGNITION
from models.config.agents_config import ARCFACE_CFG, RETINA_DETECTOR_CFG, CARD_CLASSIFICATION_MODEL
from models.config.agents_config import JOB_DESCRIPTION_RAG_EMBEDDER, JOB_DESCRIPTION_RAG_RERANKER

from models.config.agents_config import (
    # PRR
    TA_JOB_TOOLS_EXTRACTOR_CFG,
    TA_PROPOSAL_TOOLS_ANALYZER_CFG,

    RQ_REQUIREMENT_EXTRACTOR_CFG,
    RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG,

    JD_JOB_KEY_POINTS_CFG,
    JD_JOB_UNDERSTANDING_EVALUATOR_CFG,

    LANGUAGE_CLARITY_EVALUATOR_CFG,
    
    EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG,

    PRR_SUPER_AGENT_CFG,

    # PA
    PROFILE_VISUAL_BRAND_CFG,
    PROFILE_BIO_ANALYSIS_CFG,
    PROFILE_NUMERICAL_ANALYSIS_CFG,
    PROFILE_SKILLS_ANALYSIS_CFG,
    PA_PROFILE_SUPER_AGENT_CFG
)


# ---------------------------------------- CFG ---------------------------------- #
ProposalRejectionReasons_Type: TypeAlias = Any
ProfileScorer_Type: TypeAlias = Any


settings = get_settings()


def get_torch_device(device: str = "auto"):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


# ----------------- ID Reco --------------------
def get_identity_recognizer():
    import torch
    from agents.identity_recognition import FaceRecognizerArcFace

    device = get_torch_device(ARCFACE_CFG["device"])
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
    loaded = torch.load(weights_path, map_location = device)
    model.load_state_dict(loaded['model_state_dict'])

    model.eval()

    return model


def get_retina_face_detector():
    from retinaface.pre_trained_models import get_model as get_retina_model

    backbone_model = "resnet50_2020-07-20"
    device = str(get_torch_device(RETINA_DETECTOR_CFG["device"]))

    with suppress_model_loader_output():
        retina_face_detector = get_retina_model(
            model_name = backbone_model,
            max_size = RETINA_DETECTOR_CFG["max_size"],
            device = device
        )

    retina_face_detector.eval()

    return retina_face_detector



def get_card_classifier():
    return pipeline(
        task = "zero-shot-image-classification",
        model = CARD_CLASSIFICATION_MODEL
    )

# ----------------- JD ENH --------------------
def get_weaviate_client():
    """
    Returns:
        client: the Weaviate API required to use the database
    """
    import weaviate
    from weaviate.classes.init import Auth, AdditionalConfig, Timeout

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url = settings.WEAVIATE_URL,
        auth_credentials = Auth.api_key(api_key = settings.WEAVIATE_API_KEY),
        additional_config = AdditionalConfig(
            timeout = Timeout(init = 30, query = 60, insert = 120)
        )
    )

    return client


def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    with suppress_model_loader_output():
        return HuggingFaceEmbeddings(
            model_name    = JOB_DESCRIPTION_RAG_EMBEDDER["model_name"],
            model_kwargs  = JOB_DESCRIPTION_RAG_EMBEDDER["model_kwargs"],
            encode_kwargs = JOB_DESCRIPTION_RAG_EMBEDDER["encode_kwargs"],
            show_progress = False
        )

def get_raranker():
    from sentence_transformers import CrossEncoder

    with suppress_model_loader_output():
        return CrossEncoder(JOB_DESCRIPTION_RAG_RERANKER)



def get_weaviate_collection(client):
    from controllers.weaviate_controller import WeaviateController

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



# ----------- PRR -----------------
# Tools Alignment (TA)

def load_proposal_rejection_reasons_agents() -> dict[str, ProposalRejectionReasons_Type]:
    from agents.proposal_rejection_reasons.groq_core import GroqModelsAPI
    from agents.proposal_rejection_reasons import (
        JobToolsExtractor,
        ProposalToolsAnalyzer,
        JobKeyPointsExtractor,
        JobUnderstandingEvaluator,
        JobRequirementsMatcher,
        JobRequirementsExtractor,
        LanguageClarityEvaluator,
        ExperienceEvidenceAgent,
        PRR_SuperAgent
    )

    groq_client = GroqModelsAPI(api_key = settings.GROQ_API_KEY)

    agents = {}

    agents["job_tools_extractor"]           = JobToolsExtractor(groq_client = groq_client, **TA_JOB_TOOLS_EXTRACTOR_CFG)
    agents["proposal_tools_analyzer"]       = ProposalToolsAnalyzer(groq_client = groq_client, **TA_PROPOSAL_TOOLS_ANALYZER_CFG)
    agents["requirement_extractor"]         = JobRequirementsExtractor(groq_client = groq_client, **RQ_REQUIREMENT_EXTRACTOR_CFG)
    agents["requirement_matcher"]           = JobRequirementsMatcher(groq_client = groq_client, **RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG)
    agents["job_key_points_extractor"]      = JobKeyPointsExtractor(groq_client = groq_client, **JD_JOB_KEY_POINTS_CFG)
    agents["job_understanding_evaluator"]   = JobUnderstandingEvaluator(groq_client = groq_client, **JD_JOB_UNDERSTANDING_EVALUATOR_CFG)
    agents["experience_evidence_evaluator"] = ExperienceEvidenceAgent(groq_client = groq_client, **EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG)
    agents["language_clarity_evaluator"]    = LanguageClarityEvaluator(groq_client = groq_client, **LANGUAGE_CLARITY_EVALUATOR_CFG)
    agents["super_agent"]                   = PRR_SuperAgent(groq_client = groq_client, **PRR_SUPER_AGENT_CFG)

    return agents




# ----------- Profile Analysis -------------


def load_profile_scorer_agents() -> dict[str, ProfileScorer_Type]:
    """
    Initializes and returns all the sub-agents and the master orchestrator
    dedicated for the Profile Scorer pipeline.
    """
    from agents.profile_analysis import (
        VisualBrandEvaluator,
        BioAnalyzer,
        SkillsAnalyzer,
        NumericalAnalyzer,
        PA_SuperAgent
    )

    agents = {}
    agents["numerical_analyzer"]     = NumericalAnalyzer(PROFILE_NUMERICAL_ANALYSIS_CFG)
    agents["visual_brand_evaluator"] = VisualBrandEvaluator(**PROFILE_VISUAL_BRAND_CFG)
    agents["bio_analyzer"]           = BioAnalyzer(**PROFILE_BIO_ANALYSIS_CFG)
    agents["skills_analyzer"]        = SkillsAnalyzer(**PROFILE_SKILLS_ANALYSIS_CFG)
    agents["profile_super_agent"]    = PA_SuperAgent(**PA_PROFILE_SUPER_AGENT_CFG)

    return agents


# ----------- Job Recommendation System -------------

def get_rs_embedding_engine():
    from sentence_transformers import SentenceTransformer
    from agents.recommendation_system import RSEmbeddingEngine

    device = get_torch_device()

    with suppress_model_loader_output():
        model = SentenceTransformer(
            JOB_DESCRIPTION_RAG_EMBEDDER["model_name"],
            device=device,
        )
    return RSEmbeddingEngine(model=model)
