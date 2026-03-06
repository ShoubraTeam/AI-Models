# ---------------------------------------------------------------------------
# Contains the code required
# - define Global Variables [embedding_model, LLMs, ...]
# - Execute the whole system into the main function
# 
# Ahmed Ragab
# ---------------------------------------------------------------------------

# Imports
import src.utils.config as CFG
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from src.vector_database import get_weaviate_client, build_collection, load_collection, retrieve
import os
from src.utils.data_preparation import get_documents




def print_separator(n_lines = 3):
    print(3 * '\n')


# Surpass Warnings
import warnings
from transformers import logging as tf_logging
warnings.filterwarnings("ignore")
tf_logging.set_verbosity_error()




load_dotenv()


def print_retrieved_docs(docs: list):
    for idx, doc in enumerate(docs, start = 1):
        score = doc[1]
        doc = doc[0]
        print(f"-------------- Document #{idx} ---------------")
        print(f">> Document:\n {doc.properties.get('job_document')}")
        print()
        print(f">> Year : {doc.properties.get('year')}")
        print()
        print(f">> Score : {score}")
        print(50 * '---')


pd.set_option('display.max_columns', None)

if __name__ == "__main__":
    print_separator()
    print("-- Starting JOB_SUGGESTION Simulation --")


    # -- get documents_df -- 
    # documents_df = get_documents(CFG.DOCUMENTS_PATH, CFG.JOBS_PATH) [initiating db]

    # -- build the db -- 
    db_client = get_weaviate_client()
    # collection = build_collection(db_client, CFG.COLLECTION_NAME, data = documents_df, embedding_model = EMBEDDINGS) [initiating db]
    collection = load_collection(db_client, CFG.COLLECTION_NAME) # [load existing db]
    

    # -- retrieve --
    test_query = """I would need an experienced AI Engineer that can perform the following:
    - Build Scalable RAG systems.
    - Work within a team.
    - Build from scratch Machine Learning models.
    """
    retrieved_docs = retrieve(
        retriever_query = test_query,
        embedding_model = CFG.EMBEDDING_MODEL,
        cross_encoder = CFG.RERANKER,
        collection = collection
    )

    print_separator()
    print("Retrieved Documents:")
    print_retrieved_docs(retrieved_docs)



    # -- close the db connection --
    db_client.close()


    print_separator()
    print("-- Finishing Simulation --")
