# ---------------------------------------------
# Import general utility functions
# ---------------------------------------------

from typing import Any
import json
from pathlib import Path

import traceback

from models.data_config import FEATURE_ALLOWED_FEATURES, JOB_DESCRIPTION_ALLOWED_TASKS
from helpers.config import get_settings



settings = get_settings()



def validate_feature_id(feature_id: str) -> bool:
    """
    Validate the given feature ID
        
    Args:
        feature_id (str)
    
    Returns:
        is_ok (bool)
    """
    if feature_id not in FEATURE_ALLOWED_FEATURES:
        return False
    
    return True

def validate_job_description_enhancement_task(task: str) -> bool:
    """
    Validate the given task
        
    Args:
        task (str)
    
    Returns:
        is_ok (bool)
    """
    if task not in JOB_DESCRIPTION_ALLOWED_TASKS:
        return False
    
    return True
    

# -------------------- Printing Utils --------------------------
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"



def format_error(error: Exception) -> dict:
    tb = traceback.extract_tb(error.__traceback__)

    last_frame = tb[-1] if tb else None

    return {
        "error_type"    : type(error).__name__,
        "error_message" : str(error),
        "error_file"    : str(Path(last_frame.filename).resolve()) if last_frame else None,
        "error_line"    : last_frame.lineno if last_frame else None,
        "error_function": last_frame.name if last_frame else None,
    }



def print_subtitle(subtitle: str):
    print()
    subtitle = f" {subtitle} ".center(50, "=")
    print(f"{BLUE}{subtitle}{RESET}")
    print()


def print_success_message(message: str):
    print(f"{GREEN}>> {message} <<{RESET}")


def print_error(error: Exception, message: str):
    sep = 50 * '='
    print(f"{RED}{sep}{RESET}")

    print(f"{RED}>> {message} <<{RESET}: ")
    
    err_json = format_error(error)
    print(json.dumps(err_json, indent = 2))
    
    print(f"{RED}{sep}{RESET}")


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