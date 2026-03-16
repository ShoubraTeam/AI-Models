# ---------------------------------------------------------------------------
# Contains
# - Global Variables [embedding_model, LLMs, ...] & Configurations
# ---------------------------------------------------------------------------

# Imports
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from pathlib import Path


# Data Paths
JOBS_PATH = Path("./data/final_data.csv")
DOCUMENTS_PATH = Path("./data/documents.csv")

# Embedding Model
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL_NAME,
    model_kwargs = {"device" : "cuda"},
    encode_kwargs = {"batch_size" : 128}
)

# Cross Encoder
RERANKER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RERANKER = CrossEncoder(RERANKER_NAME)


# Database CFG
COLLECTION_NAME = "JOB_SUGGESTION_COLLECTION_V1"


# Models
DETECTION_MODEL = "llama-3.1-8b-instant"
ENHANCEMENT_MODEL_1 = "llama-3.3-70b-versatile"
SKILLS_EXTRACTOR_MODEL = "llama-3.1-8b-instant"