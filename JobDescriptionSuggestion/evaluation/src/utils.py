# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------


import json
import argparse

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
    
    # paths
    parser.add_argument("--relevant_doc_path", type = str, help = "The path to the relevant documents")
    parser.add_argument("--eval_data_path", type = str, help = "The path to the eval data")

    # models
    parser.add_argument("--embedding_model", type = str, help = "Model to use in Embedding")
    parser.add_argument("--reranker", type = str, help = "Model to use in Reranker")
    parser.add_argument("--tools_detector", type = str, help = "Model to detect the existing tools in the given job description")
    parser.add_argument("--tools_extractor", type = str, help = "Model to extract the existing tools in the given job description")
    parser.add_argument("--job_enhnacer", type = str, help = "Model to enhance the given job description")

    # others
    parser.add_argument("--repeats", type = int, help = "Number of times to repeat the models. Used in calculating avg_time")


    args = parser.parse_args().__dict__
    return args