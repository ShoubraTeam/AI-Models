# --------------------------------------------
# Loading:
# - Agents
# - Models
# - Weaviate collection
# - Clients
# --------------------------------------------

import os

from core.startup_noise import configure_startup_noise, suppress_model_loader_output

configure_startup_noise()

import torch

#from retinaface.pre_trained_models import get_model as get_retina_model
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


# agents
from agents.proposal_rejection_reasons import (
    JobToolsExtractor,
    ProposalToolsAnalyzer,
    JobKeyPointsExtractor,
    JobUnderstandingEvaluator,
    JobRequirementsMatcher,
    JobRequirementsExtractor,
    LanguageClarityEvaluator,
    ExperienceEvidenceAgent,
    ProposalRejectionSuperAgent
)

ProposalRejectionReasons_Type = (
    JobToolsExtractor           |
    ProposalToolsAnalyzer       |
    JobKeyPointsExtractor       |
    JobUnderstandingEvaluator   |
    JobRequirementsMatcher      |
    JobRequirementsExtractor    |
    LanguageClarityEvaluator    |
    ExperienceEvidenceAgent     |
    ProposalRejectionSuperAgent
)


# prompts
from prompts import (
    JOB_TOOLS_EXTRACTION_PROMPT,
    PROPOSAL_TOOLS_EXTRACTION_PROMPT,
    EXPERIENCE_EVIDENCE_PROMPT,
    JOB_KEY_POINTS_EXTRACTION_PROMPT,
    JOB_UNDERSTANDING_EVALUATOR_PROMPT,
    REQUIREMENT_EXTRACTOR_PROMPT,
    REQUIREMENT_MATCHER_PROMPT,
    LANGUAGE_CLARITY_EVALUATOR_PROMPT,
    SUPER_AGENT_SYSTEM_PROMPT
)


# prompts
from models.pydantic_schemas import (
    JobToolResponse,
    ProposalToolsResponse,
    JobKeyPointsSchema,
    JobUnderstandingEvalSchema,
    ExtractedRequirementsSchema,
    RequirementCoverageSchema,
    ExperienceEvidenceSchema,
    LanguageClarityEvalSchema,
    SuperAgentResponse
)



settings = get_settings()


def get_torch_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


# identity recognition
def get_identity_recognizer() -> FaceRecognizerArcFace:
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
    with suppress_model_loader_output():
        return HuggingFaceEmbeddings(
            model_name    = JOB_DESCRIPTION_RAG_EMBEDDER["model_name"],
            model_kwargs  = JOB_DESCRIPTION_RAG_EMBEDDER["model_kwargs"],
            encode_kwargs = JOB_DESCRIPTION_RAG_EMBEDDER["encode_kwargs"],
            show_progress = False
        )

def get_raranker() -> CrossEncoder:
    with suppress_model_loader_output():
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

# ----------- Proposal Rejection Reasons -------------
# Tools Alignment (TA)
TA_TOOL_ALIGNMENT_THRESHOLD = 0.5
TA_JOB_TOOLS_EXTRACTOR_CFG = {
    "model_name"         : "groq:llama-3.1-8b-instant",
    "system_prompt"      : JOB_TOOLS_EXTRACTION_PROMPT,
    "structured_response": JobToolResponse,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

TA_PROPOSAL_TOOLS_ANALYZER_CFG = {
    "model_name"         : "groq:llama-3.1-8b-instant",
    "system_prompt"      : PROPOSAL_TOOLS_EXTRACTION_PROMPT,
    "structured_response": ProposalToolsResponse,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

# Job Understanding (JD)
JD_JOB_UNDERSTANDING_THRESHOLD = 0.5
JD_JOB_KEY_POINTS_CFG = {
    "model_name"         : "groq:openai/gpt-oss-120b",
    "system_prompt"      : JOB_KEY_POINTS_EXTRACTION_PROMPT,
    "structured_response": JobKeyPointsSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

JD_JOB_UNDERSTANDING_EVALUATOR_CFG = {
    "model_name"         : "groq:openai/gpt-oss-120b",
    "system_prompt"      : JOB_UNDERSTANDING_EVALUATOR_PROMPT,
    "structured_response": JobUnderstandingEvalSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

# Requirement Coverage (RQ)
RQ_REQUIREMENT_COVERAGE_THRESHOLD = 0.5
RQ_REQUIREMENT_EXTRACTOR_CFG = {
    "model_name"         : "groq:llama-3.1-8b-instant",
    "system_prompt"      : REQUIREMENT_EXTRACTOR_PROMPT,
    "structured_response": ExtractedRequirementsSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG = {
    "model_name"         : "groq:openai/gpt-oss-120b",
    "system_prompt"      : REQUIREMENT_MATCHER_PROMPT,
    "structured_response": RequirementCoverageSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

# Language Clarity
LANGUAGE_CLARITY_THRESHOLD = 0.5 
LANGUAGE_CLARITY_EVALUATOR_CFG = {
    "model_name"         : "groq:openai/gpt-oss-120b",
    "system_prompt"      : LANGUAGE_CLARITY_EVALUATOR_PROMPT,
    "structured_response": LanguageClarityEvalSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

# Evidence of experience
EXPERIENCE_EVIDENCE_THRESHOLD = 0.5
EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG = {
    "model_name"         : "groq:openai/gpt-oss-120b",
    "system_prompt"      : EXPERIENCE_EVIDENCE_PROMPT,
    "structured_response": ExperienceEvidenceSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

# Super Agent
SUPER_AGENT_CFG = {
    "model_name"         : "groq:openai/gpt-oss-120b",
    "system_prompt"      : SUPER_AGENT_SYSTEM_PROMPT,
    "structured_response": SuperAgentResponse,
    "temperature"        : 0.1,
    "max_tokens"         : 1024,
    "top_p"              : 0.9
}

def load_proposal_rejection_reasons_agents() -> dict[str, ProposalRejectionReasons_Type]:
    agents = {}

    agents["job_tools_extractor"]           = JobToolsExtractor(**TA_JOB_TOOLS_EXTRACTOR_CFG)
    agents["proposal_tools_analyzer"]       = ProposalToolsAnalyzer(**TA_PROPOSAL_TOOLS_ANALYZER_CFG)
    agents["requirement_extractor"]         = JobRequirementsExtractor(**RQ_REQUIREMENT_EXTRACTOR_CFG)
    agents["requirement_matcher"]           = JobRequirementsMatcher(**RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG)
    agents["job_key_points_extractor"]      = JobKeyPointsExtractor(**JD_JOB_KEY_POINTS_CFG)
    agents["job_understanding_evaluator"]   = JobUnderstandingEvaluator(**JD_JOB_UNDERSTANDING_EVALUATOR_CFG)
    agents["experience_evidence_evaluator"] = ExperienceEvidenceAgent(**EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG)
    agents["language_clarity_evaluator"]    = LanguageClarityEvaluator(**LANGUAGE_CLARITY_EVALUATOR_CFG)
    agents["super_agent"]                   = ProposalRejectionSuperAgent(**SUPER_AGENT_CFG)

    return agents


# ----------- Job Recommendation System -------------

def get_rs_embedding_engine():
    from agents.recommendation_system import RSEmbeddingEngine
    from sentence_transformers import SentenceTransformer   # ← add this
    import torch                                            # ← add this

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with suppress_model_loader_output():
        model = SentenceTransformer(
            JOB_DESCRIPTION_RAG_EMBEDDER["model_name"],
            device=device,
        )
    return RSEmbeddingEngine(model=model)