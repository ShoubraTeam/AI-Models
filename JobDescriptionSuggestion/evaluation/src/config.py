# ------------------------------------------------------------
# Contains Configuration for the Evaluation System
# ------------------------------------------------------------

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



EMBEDDING_BATCH_SIZE = 128 if str(DEVICE) == "cuda" else 8
BGE_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
NOMIC_EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

BGE_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MINILM_RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
MIXEDBREAD_RERANKER_MODEL_NAME = 'mixedbread-ai/mxbai-rerank-large-v1'

EVAL_COLLECTION_NAME = "eval_collection"