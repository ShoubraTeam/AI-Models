# --------------------------------------------
# Import Utility Functions
# --------------------------------------------
import json 
from typing import Any
from helpers.config import BLUE, RESET, GREEN, RED
from argparse import ArgumentParser
from .config import EVALUATION_MODELS_MAPPING

# Printing Utilities
def print_subtitle(subtitle: str):
    print()
    subtitle = f" {subtitle} ".center(50, "=")
    print(f"{BLUE}{subtitle}{RESET}")
    print()


def print_success_message(message: str):
    print(f"{GREEN}>> {message} <<{RESET}")

def print_error_message(message: str):
    print(f"{RED}>> {message} <<{RESET}")

def print_title(title: str, n_sep: int =  100, sep: str = "="):
    """Printing a title in a well-formatted manner"""
    title = f" {title} ".center(n_sep, sep)
    print(title)


def print_structured_response(structured_response):
    """Printing the Agent Structured Response"""
    # print_title("Structured Response", 50)
    structured_response = structured_response.model_dump()

    for attb, value in structured_response.items():
        print()
        print(f"{attb.capitalize()}", end = "")

        if isinstance(value, list):
            print(":")
            for item in value:
                if isinstance(item, dict): 
                    for key, val in item.items():
                        print(f"\t{key} => {val}")
                else:
                    print(f"\t{item}")
                
                print()

        else:
            print(f"=> {value}")


def print_dict(dic: dict):
    print()
    for k, v in dic.items():
        print(f"{k} => {v}")
    
def print_semi_dict(semi_dict: Any):
    print()
    items = semi_dict.__dict__.items()
    for k, v in items:
        print(f"{k} => {v}")

    

def print_data(data: Any):
    if isinstance(data, list):
        if isinstance(data[0], dict):
            for dic in data:
                print_dict(dic)
        
        elif hasattr(data[0], "__dict__"):
            for dic in data:
                print_semi_dict(dic)
        
        else:
            print()
            print(data)

    elif isinstance(data, dict):
        print_dict(data)
    
    elif hasattr(data, "__dict__"):
        print_semi_dict(data)
    
    else:
        print()
        print(data)
    


def print_eval_data(eval_data: list[dict]):
    print_subtitle("Extracted Eval Data:")
    
    for idx, sample in enumerate(eval_data, start = 1):
        print(f"Sample #{idx}")

        job_data = sample["job"]
        for k, v in job_data.items():
            print(f">> {k}:\n{v}\n\n")

        proposals = sample["proposals"]
        for proposal_idx, proposal in enumerate(proposals, start = 1):
            print(f"Proposal #{proposal_idx}")

            for k, v in proposal.items():
                print(f">> {k}:\n{v}\n\n")
            
        print("------")


# Loading data
def load_file(file_path: str):
    with open(file_path, mode = "r", encoding = "utf-8") as f:
        return f.read()


def load_json(file_path: str) -> dict:
    with open(file_path, mode = "r", encoding = "utf-8") as f:
        return json.load(f)
    

# accepting terminal arguments
def get_terminal_arguments():
    """
    Return dict of given arguments
    """

    parser = ArgumentParser(description = "Parsing Terminal Arguments")
    

    parser.add_argument(
        "--task", 
        type = str,
        help = "The task to evaluate, tools_alignment, job_understanding, etc..."
    )

    parser.add_argument(
        "--rounds",
        type = int,
        help = "The number of times the evaluation process should run.",
        default = 5
    )


    parser.add_argument(
        "--models",
        type = str,
        nargs = "+",
        help = "The model used in evaluating. Assuming `model_1` for extraction, `model_2` for matching, ..."
    )

    parser.add_argument(
        "--temperature",
        type = float,
        help = "The temperature that controls the model's randomness."
    )

    parser.add_argument(
        "--max_tokens",
        type = int,
        help = "The max_tokens that the model should generate."
    )


    parser.add_argument(
        "--top_p",
        type = float,
        help = "The top_p that control's selected tokens"
    )



    # files
    parser.add_argument(
        "--eval_data_path",
        type = str,
        help = "The path to the evaluation data."
    )

    parser.add_argument(
        "--output_path",
        type = str,
        help = "The path to output the results in."
    )


    args = parser.parse_args().__dict__
    return args

def parse_terminal_arguments():
    """
    Accepting the terminal arguments & parse them
    """

    args = get_terminal_arguments()


    # parse models
    models = args["models"]
    args["models"] = [
        EVALUATION_MODELS_MAPPING[model]
        for model in models
    ]

    # parse model args
    kwargs = {}
    for arg in ["temperature", "max_tokens", "top_p"]:
        value = args.pop(arg, None)

        if value is not None:
            kwargs[arg] = value
    
    args["model_kwargs"] = kwargs

    return args