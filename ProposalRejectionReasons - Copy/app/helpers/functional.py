# --------------------------------------------
# Import Utility Functions
# --------------------------------------------
import json 
from typing import Any
from helpers.config import BLUE, RESET, GREEN, RED

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


def print_dict(dic: dict, n_identation = 0):
    identation = ""
    if n_identation > 0: identation = "\t"*n_identation

    print()
    for k, v in dic.items():
        print(f"{identation}{k} => {v}")
    
def print_semi_dict(semi_dict: Any, n_identation = 0):
    identation = ""
    if n_identation > 0: identation = "\t"*n_identation

    print()
    items = semi_dict.__dict__.items()
    for k, v in items:
        print(f"{identation}{k} => {v}")

    

def print_data(data: Any, n_identation = 0):
    identation = ""
    if n_identation > 0: identation = "\t"*n_identation

    if isinstance(data, list):
        if isinstance(data[0], dict):
            for dic in data:
                print_dict(dic, n_identation)
        
        elif hasattr(data[0], "__dict__"):
            for dic in data:
                print_semi_dict(dic, n_identation)
        
        else:
            print()
            print(f"{identation}{data}")

    elif isinstance(data, dict):
        print_dict(data, n_identation)
    
    elif hasattr(data, "__dict__"):
        print_semi_dict(data, n_identation)
    
    else:
        print()
        print(f"{identation}{data}")
    


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
    
