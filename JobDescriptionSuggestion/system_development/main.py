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
from src.job_enhancer import Enhancer
from src.vector_database import get_weaviate_client, build_collection



def print_separator(n_sep = 100):
    print('-' * n_sep)


# Surpass Warnings
import warnings
from transformers import logging as tf_logging
warnings.filterwarnings("ignore")
tf_logging.set_verbosity_error()

load_dotenv()


TEST_CASES = [
    {
        "title" : "AI Engineer",
        "description" : """I would need an experienced AI Engineer that can perform the following:
        - Build Scalable RAG systems.
        - Work within a team.
        - Build from scratch Machine Learning models.
        """
    },
    # {
    #     "title" : "Backend Engineer",
    #     "description" : """I would need an experienced web developer that can build an e-commerce website.
    #     """
    # },
    # {
    #     "title" : "Backend Engineer",
    #     "description" : """I would need an experienced web developer that can build an e-commerce website. He should be proficient in Node.JS & NoSQL databases.
    #     """
    # },
]


pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    print_separator()
    print("-- Starting JOB_SUGGESTION Simulation --")

    # -- load documents_df --
    documents_df = pd.read_parquet("data\documents_df.parquet")


    # -- build the db -- 
    # db_client = get_weaviate_client()
    # collection = build_collection(db_client, CFG.COLLECTION_NAME, data = documents_df) 
    

    # # -- Test -- 
    enhnacer = Enhancer(
        enhancement_model = CFG.ENHANCEMENT_MODEL_1,
        detection_model = CFG.DETECTION_MODEL,
        skills_extractor = CFG.SKILLS_EXTRACTOR_MODEL,
        collection_name = CFG.COLLECTION_NAME,
        model_provider = "groq"
    )
    

    for test_case in TEST_CASES:
        response = enhnacer.enhnace(
            job_info = test_case,
            temperature = 0.7,
            max_tokens = 512,
            top_p = 0.8,
        )

        print(response)
        print_separator()

        
    enhnacer.close_db()
