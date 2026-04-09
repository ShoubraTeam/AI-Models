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





# ------------------------------------------------------------------------------------------------
# Prompts & LLMs & LLMs Errors

TOOLS_DETECTOR_PROMPT = """You are an expert HR assistant. 
Your task is to analyze the provided job description and determine if it explicitly contains specific tools/frameworks that is related to the job.
Respond ONLY with 'Yes' if it contains skills, and 'No' if it does not contain any skills. Do not provide any further explanation.
Examples
- Job Description: I seek for an experienced AI Engineer who can build a customer support chatbot.
  Response: No

- Job Description: I seek for an experienced AI Engineer who can build a customer support chatbot. He should be able to use Python and vector databases, and build RAG systems.
  Response: Yes
"""


TOOLS_EXTRACTOR_PROMPT = """You are an experienced text analyzer. You will be given a job description. Your role is to extract only the tools/frameworks mentioned in the description.
Your output should be in the form of a list of tools/frameworks. In particular, you should output the following:
[
    'tool_1',
    'tool_2',
    'tool_3',
    .
    .
    .
    'tool_n'
]

Instructions
- Only output the existing tools.
- In your response, only include the tools list. Do not add any other tokens to the list.
- Add the list braces '[' and ']' before and after the list.
- Add single quotes ' before and after each tool.
"""

# LLMs
LLAMA_DETECTION_MODEL = "llama-3.1-8b-instant"
LLAMA_EXTRACTOR_MODEL = "llama-3.1-8b-instant"
LLAMA_ENHANCEMENT_MODEL = "llama-3.3-70b-versatile"
LLAMA_JUDGE_MODEL = "llama-3.3-70b-versatile"

# Errors
TOOLS_DETECTOR_ERROR_OUTPUT = -1

