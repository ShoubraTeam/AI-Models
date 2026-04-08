# ------------------------------------------------------------
# Main Script for Evaluating the Models
# ------------------------------------------------------------


from src.evaluate_rag import *
import src.config as CFG
import argparse
import json
from src.vector_database import *
from dotenv import load_dotenv

load_dotenv()

def print_title(title: str, n_sep = 150):
    print()
    title = f" {title} "
    print(title.center(n_sep, "="))


def load_data(file_path, file_type = "json"):
    if file_type == "json":
        with open(file_path, mode = 'r', encoding = 'utf-8') as f:
            return json.load(f)
        

def parse_arguments():
    """Return Dict of Terminal Arguments"""
    parser = argparse.ArgumentParser(description = "Parsing Arguments Dynamically")
    parser.add_argument("--relevant_doc_path", type = str, help = "The path to the relevant documents")
    parser.add_argument("--eval_data_path", type = str, help = "The path to the eval data")
    parser.add_argument("--embedding_model", type = str, help = "Model to use in Embedding")
    parser.add_argument("--reranker", type = str, help = "Model to use in Reranker")
    parser.add_argument("--repeats", type = int, help = "Number of times to repeat the models. Used in calculating avg_time")
    args = parser.parse_args().__dict__

    return args

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

    


    client.close()