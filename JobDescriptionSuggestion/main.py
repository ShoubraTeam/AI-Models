# ---------------------------------------------------------------------------
# Contains the code required
# - define Global Variables [embedding_model, LLMs, ...]
# - Execute the whole system into the main function
# 
# Ahmed Ragab
# ---------------------------------------------------------------------------

# Imports
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from src.vector_database import get_weaviate_client, build_collection, load_collection, retrieve_documents
import os
from src.utils.data_preparation import get_documents
from langchain_huggingface import HuggingFaceEmbeddings

import warnings
import logging
warnings.filterwarnings("ignore")
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

def print_separator(n_lines = 3):
    print(3 * '\n')


# CFG
JOBS_PATH = Path("./data/final_data.csv")
DOCUMENTS_PATH = Path("./data/documents.csv")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "JOB_SUGGESTION_COLLECTION_V1"
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL,
    model_kwargs = {"device" : "cuda"},
    encode_kwargs = {"batch_size" : 128}
)


load_dotenv()


def print_retrieved_objects(objects: list):
    for idx, obj in enumerate(objects, start = 1):
        print(f"-------------- Document #{idx} ---------------")
        print(f">> Document:\n {obj.properties.get('job_document')}")
        print()
        print(f">> Year : {obj.properties.get('year')}")
        print(50 * '---')


pd.set_option('display.max_columns', None)
if __name__ == "__main__":
    print_separator()
    print("-- Starting JOB_SUGGESTION Simulation --")


    # -- get documents_df -- 
    # documents_df = get_documents(DOCUMENTS_PATH, JOBS_PATH) [initiating db]

    # -- build the db -- 
    db_client = get_weaviate_client()
    # collection = build_collection(db_client, COLLECTION_NAME, data = documents_df, embedding_model = EMBEDDINGS) [initiating db]
    collection = load_collection(db_client, COLLECTION_NAME) # [load existing db]
    

    # -- retrieve --
    test_query = """I would need an experienced AI Engineer that can perform the following:
    - Build Scalable RAG systems.
    - Work within a team.
    - Build from scratch Machine Learning models.
    """
    retrieved_doc = retrieve_documents(
        query = test_query,
        collection = collection,
        embedding_model = EMBEDDINGS
    )

    print_separator()
    print("Retrieved Documents:")
    print_retrieved_objects(retrieved_doc)



    # -- close the db connection --
    db_client.close()


    print_separator()
    print("-- Finishing Simulation --")
