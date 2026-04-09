# ------------------------------------------------------------
# Main Script for Evaluating the Models
# ------------------------------------------------------------


from src.evaluate_rag import *
import src.config as CFG
from src.vector_database import *
from dotenv import load_dotenv
from src.utils import *
from src.evaluate_llms import evaluate_llms

load_dotenv()

import warnings
import os

# Suppress all Python warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow/other C++ backend logs (if applicable)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import logging as transformers_logging

# This hides the LOAD REPORT and "Some weights were not initialized" messages
transformers_logging.set_verbosity_error()

from langchain_groq import ChatGroq
judge_llm = ChatGroq(model = CFG.LLAMA_JUDGE_MODEL)

from langchain_huggingface import HuggingFaceEmbeddings
judge_embeddings = HuggingFaceEmbeddings(model_name = CFG.BGE_EMBEDDING_MODEL_NAME)


if __name__ == "__main__":
    # Parsing Arguments
    print_title("Parsing Arguments")
    args = parse_arguments()
    print(f">> Arguments: {args}")
    client = get_weaviate_client()

    # loading data
    print_title("Loading Data")
    relevant_doc = load_data(args["relevant_doc_path"])
    eval_data = load_data(args["eval_data_path"])
    print(f">> Data Loaded:")
    print(f"\tN.Records in Eval Data: {len(eval_data)}")
    print(f"\tN.Relevant Documents  : {len(relevant_doc)}")

    # Embedding
    print_title("Embedding")
    print(">> Loading Models:\n")
    embedding_model = load_embedding_model(args["embedding_model"])
    reranker = load_reranker(model_name = args['reranker'])
    
    
    print(f"\n>> Embedding {len(relevant_doc)} Documents:")
    avg_time = get_embedding_time(embedding_model, chunks = [doc['chunk'] for doc in relevant_doc], repeats = args["repeats"])
    print(f"\tAVG Embedding Time: {avg_time:.4f} Seconds.")
    

    print_title("Building Vector Database")
    collection = prepare_collection(client, CFG.EVAL_COLLECTION_NAME, embedding_model = embedding_model, chunks = relevant_doc)


    print_title("Evaluating retreiver")
    retreival_results = evaluate_retreival_operation(
        eval_data = eval_data,
        embedding_model = embedding_model,
        reranker = reranker,
        collection = collection,
        reranker_name = args["reranker"]
    )

    print(">> Results:")
    print(f"\tAvg Recall@K   : {retreival_results['avg_recall_at_k_score']:0.4f}")
    print(f"\tAvg Precision@K: {retreival_results['avg_precision_at_k_score']:0.4f}")
    print(f"\tAvg MRR        : {retreival_results['average_mean_reciporcal_rank']:0.4f}")

    

    # LLMs
    print_title("Evaluating LLMs")
    results = evaluate_llms(
        eval_data = eval_data,
        tools_detector = CFG.LLAMA_DETECTION_MODEL,
        tools_extractor = CFG.LLAMA_EXTRACTOR_MODEL,
        job_enhnacer = CFG.LLAMA_ENHANCEMENT_MODEL,
        judge_llm = judge_llm,
        judge_embeddings = judge_embeddings,
        temperature = 0.7,
    )

    print(results)


    client.close()