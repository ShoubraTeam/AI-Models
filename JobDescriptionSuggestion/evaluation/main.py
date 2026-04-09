# ------------------------------------------------------------
# Main Script for Evaluating the Models
# ------------------------------------------------------------


from src.evaluate_rag import *
import src.config as CFG
from src.vector_database import *
from dotenv import load_dotenv
from src.utils import *
from src.evaluate_llms import evaluate_llms





# ------------------------------------------------------------------------------------------
# Suppress warnings
import warnings
import os
from transformers import logging as transformers_logging
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
transformers_logging.set_verbosity_error()
# ------------------------------------------------------------------------------------------

import csv

def log_event(file, message, title = False, dic = False, special = False):
    file.write(f"\n")
    if title:
        message = f" {message} "
        file.write(message.center(100, "-"))
    
    elif dic:
        for key, val in message.items():
            if special:
                file.write(f">> {key}: {val}\n")
            else:
                file.write(f'- {key} --> {val}\n')
    
    else:
        file.write(message)

    file.flush()

def save_result(type = "", embedder = "", reranker = "", detector = "", extractor = "", enhancer = "", metric = "", value = ""):
    csv_writer.writerow([type, embedder, reranker, detector, extractor, enhancer, metric, value])
    csv_file.flush() 


load_dotenv()
if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------
    # Setup
    args = parse_arguments()
    print_title(f"Run #{args['run_id']}")

    log_file = open(file = args["log_file"], mode = "a", encoding = "utf-8")
    csv_file = open(file = args["csv_file"], mode = "a", newline = "", encoding = "utf-8")
    csv_writer = csv.writer(csv_file)

    log_event(file = log_file, message = f"Current Run ID: {args['run_id']}", title = True)
    # ----------------------------------------------------------------------------------------------------
    # Loading Data
    log_event(file = log_file, message = "Loading Global Variables & Data", title = True)    
    relevant_doc = load_data(args["relevant_doc_path"])
    eval_data = load_data(args["eval_data_path"])
    log_event(file = log_file, message = {"Number of Samples in Eval Data" : len(eval_data), "Number of Relevant Documents" : len(relevant_doc)}, dic = True)

    # ----------------------------------------------------------------------------------------------------
    # RAG
    if args["component"] == "RAG":
        log_event(file = log_file, message = "Embedding & Constructing Vector Database", title = True)
        client = get_weaviate_client()

        embedding_model = load_embedding_model(args["embedding_model"])
        reranker = load_reranker(model_name = args['reranker'])
        avg_emb_time = get_embedding_time(embedding_model, chunks = [doc['chunk'] for doc in relevant_doc], repeats = args["repeats"])
        collection = prepare_collection(client, CFG.EVAL_COLLECTION_NAME, embedding_model = embedding_model, chunks = relevant_doc)

    

        log_event(file = log_file, message = "Evaluating Retreival Process", title = True)

        print("- Embedder : " + args["embedding_model"])
        print("- Reranker: " + args["reranker"])

        retreival_results = evaluate_retreival_operation(
            eval_data = eval_data,
            embedding_model = embedding_model,
            reranker = reranker,
            collection = collection,
        )

        log_event(
            file = log_file,
            message = {"Embedder" : args["embedding_model"], "Rerakner" : args["reranker"]},
            dic = True,
            special = True
        )

        log_event(file = log_file, message = {"Embedding Time" : round(avg_emb_time, 3), "Number of Embedded Documents" : len(relevant_doc)}, dic = True)
        log_event(file = log_file,
            message = {
                "Average Recall@k": round(retreival_results['recall@k'], 3),
                "Average Precision@k": round(retreival_results['precision@k'], 3),
                "Average MRR": round(retreival_results['mrr'], 3),
            },
            dic = True
        )

        save_result(
            type = "RAG",
            embedder = args["embedding_model"],
            metric = "embedding_time",
            value = avg_emb_time
        )

        for metric, value in retreival_results.items():
            save_result(
                type = "RAG",
                embedder = args["embedding_model"],
                reranker = args["reranker"],
                metric = metric,
                value = round(value, 4)
            )   

    # ----------------------------------------------------------------------------------------------------
    # LLMs
    elif args['component'] == "LLM":
        log_event(file = log_file, message = "Evaluating LLMs", title = True)
        llm_judge, embedding_judge = load_judges()
        client_name = "sambanova" if args["enhancer"] == "deepseek" else "groq"

        print("- Detector : " + args["detector"])
        print("- Extractor: " + args["extractor"])
        print("- Enhancer : " + args["enhancer"])

        llms_results = evaluate_llms(
            client_name = "groq",
            eval_data = eval_data,
            tools_detector = CFG.MODELS_DICT[args["detector"]],
            tools_extractor = CFG.MODELS_DICT[args["extractor"]],
            job_enhnacer = CFG.MODELS_DICT[args["enhancer"]],
            judge_llm = llm_judge,
            judge_embeddings = embedding_judge,
            temperature = 0.7,
        )

        log_event(
            file = log_file,
            message = {"Detector" : args["detector"], "Extractor" : args["extractor"], "Enhancer" : args["enhancer"]},
            dic = True,
            special = True
        )


        log_event(file = log_file, message = llms_results, dic = True)

        for metric, value in llms_results.items():
            save_result(
                type = "LLM",
                detector = args["detector"],
                extractor = args["extractor"],
                enhancer = args["enhancer"],
                metric = metric,
                value = value
            )

    # ----------------------------------------------------------------------------------------------------
    # Finishing
    log_event(file = log_file, message = 150 * '=')
    log_file.close()
    csv_file.close()
    # client.close()
    # ----------------------------------------------------------------------------------------------------