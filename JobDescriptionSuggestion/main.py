# ---------------------------------------------------------------------------
# Contains the code required
# - define Global Variables [embedding_model, LLMs, ...]
# - Execute the whole system into the main function
# 
# Ahmed Ragab
# ---------------------------------------------------------------------------

# Imports
import src.utils.config as CFG
import pandas as pd
from dotenv import load_dotenv
from src.utils.data_preparation import get_documents
from src.job_enhancer import Enhancer



def print_separator(n_sep = 100):
    print('-' * n_sep)


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
    # documents_df = get_documents(CFG.DOCUMENTS_PATH, CFG.JOBS_PATH)

    # -- build the db -- 
    # db_client = get_weaviate_client()
    # collection = build_collection(db_client, CFG.COLLECTION_NAME, data = documents_df, embedding_model = EMBEDDINGS) [initiating db]
    

    # -- Test -- 
    enhnacer = Enhancer(
        enhancement_model = CFG.ENHANCEMENT_MODEL_1,
        detection_model = CFG.DETECTION_MODEL,
        collection_name = CFG.COLLECTION_NAME,
        model_provider = "groq"
    )

    test_query = """I would need an experienced AI Engineer that can perform the following:
    - Build Scalable RAG systems.
    - Work within a team.
    - Build from scratch Machine Learning models.
    """
    job_info = {
        "title" : "AI Engineer",
        "description" : test_query,
    }


    enhanced, formatted_job = enhnacer.enhance_old_desc(
        job_info = job_info,
        use_rag = True,
        temperature = 0.7,
        max_tokens = 1000
    )
    
    print_separator()
    print(enhanced)
    print_separator()
    print("Original\n")
    print(formatted_job)


    enhnacer.close_db()
