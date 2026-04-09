# ------------------------------------------------------------
# Load & Evaluate the Different LLMs in the pipeline
# ------------------------------------------------------------

# ---------------------------------------------------------------------------
# Contains the required functions to connect to
# - Connect to GROQ
# - Using GROQ models to enhance the the old job desc
# - Using GROQ models to detect if the old desc contains skills or not
# ---------------------------------------------------------------------------
 

import pandas as pd
from groq import Groq
import src.config as CFG
import ast
from ragas import evaluate, RunConfig
from ragas.metrics import _answer_similarity
from datasets import Dataset
from sambanova import SambaNova
import os
import time
# ------------------------------------------------------------------------------------------------
# Accessing Models

def get_groq_client():
    """
    Returns:
        client: the GROQ API required to use the model
    """
    client = Groq()
    return client

def get_sambanova_client():
    client = SambaNova(
        base_url = "https://api.sambanova.ai/v1",
        api_key = os.getenv("SAMBANOVA_API_KEY")
    )

    return client


def query_model(
    client,
    query: str,
    model_name: str,
    system_prompt: str,
    **kwargs
):
    """Prompting a GROQ/Sampanova Model for any of [detecting tools -- extracting tools -- enhance a job]"""
    messages = [
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : query}
    ]

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model = model_name,
        messages = messages,
        stream = False,
        **kwargs
    )
    inference_time = time.perf_counter() - start_time

    model_output = response.choices[0].message.content

    return model_output, inference_time
# ------------------------------------------------------------------------------------------------
# Evaluating the Models

def evaluate_tools_detector(true: bool, model_output: str):
    """
    Evaluating the tool detector model

    Args:   
        true        : the ground truth
        model_output: the detector's raw output
    """
    if model_output.lower() in ["yes", "no"]:
        pred = "yes" in model_output.lower()
        if true == pred:
            return 1
        else:
            return 0
    else:
        return CFG.TOOLS_DETECTOR_ERROR_OUTPUT
    
    
def evaluate_tools_extractor(true_tools: list, model_output: str):
    """
    Evaluating the tool detector model

    Args:   
        true_tools     : the ground truth list of tools
        model_output   : the extractor's raw output

    Returns:
        tuple of:
            precision (float): (TP) / (TP + FP)
            recall (float)   : (TP) / (TP + FN)
            f1 score (float) : balance between recall & precision
    """
    try:
        extracted_tools = list(set(ast.literal_eval(model_output)))[:10]
        extracted_tools = set([tool.lower() for tool in extracted_tools])
    except Exception as e:
        raise Exception(f"The extractor output cannot be parsed as list: {e}")

    true_tools = set([tool.lower() for tool in true_tools])
    
    TP = len(extracted_tools.intersection(true_tools))
    FP = len(extracted_tools - true_tools)
    FN = len(true_tools - extracted_tools)

    if TP == 0:
        return 0, 0, 0, extracted_tools
  
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
  
    return precision, recall, f1, extracted_tools
    
# ------------------------------------------------------------------------------------------------
# Run Evaluation

def evaluate_llms(
    client_name: str,
    eval_data: list,
    tools_detector: str,
    tools_extractor: str,
    job_enhnacer: str,
    judge_llm,
    judge_embeddings,
    **kwargs
):
    if client_name == "groq":
        client = get_groq_client()
    elif client_name == "sambanova":
        client = get_sambanova_client()
    else:
        raise ValueError("Invalid Client Name")
    

    client = get_groq_client()
    enhancement_client = client if client_name == 'groq' else get_sambanova_client()

    # detector metrics
    detector_time = 0
    detector_acc = 0

    # extractor metrics
    extractor_time = 0
    extractor_pre = 0
    extractor_rec = 0
    extractor_f1 = 0
    n_jobs_has_tools = 0

    # enhancer metrics
    ragas_dataset = {
        "question" : [],
        "ground_truth" : [],
        "answer" : [],
    }
    enhancer_time = 0

    for sample in eval_data:
        extracted_tools = []

        # evaluate tools detector
        model_output, inference_time = query_model(
            client = client,
            query = sample["original_job_description"],
            model_name = tools_detector,
            system_prompt = CFG.TOOLS_DETECTOR_PROMPT,
            max_tokens = 1,
            **kwargs
        )
        detector_time += inference_time
        detector_acc += 1 if evaluate_tools_detector(true = sample["has_tools"], model_output = model_output) == 1 else 0

        # evaluate tools extractor
        if sample["has_tools"]:
            n_jobs_has_tools += 1
            model_output, inference_time = query_model(
                client = client,
                query = sample["original_job_description"],
                model_name = tools_extractor,
                system_prompt = CFG.TOOLS_EXTRACTOR_PROMPT,
                temperature = 0.3
            )
            extractor_time += inference_time
            extractor_results = evaluate_tools_extractor(true_tools = sample["client_tools"], model_output = model_output)
            extractor_pre += extractor_results[0]
            extractor_rec += extractor_results[1]
            extractor_f1 += extractor_results[2]
            extracted_tools = extractor_results[3]

        
        # evaluate the enhancer
        question, model_output, inference_time = construct_ragas_question(
            enhancement_client, 
            job_enhnacer, 
            sample["original_job_description"], 
            extracted_tools, 
            **kwargs
        )

        ragas_dataset["question"].append(question)
        ragas_dataset["ground_truth"].append(sample["enhanced_job_description"])
        ragas_dataset["answer"].append(model_output)
        enhancer_time += inference_time

    # evaluate ragas
    ragas_dataset = Dataset.from_dict(ragas_dataset)
    ragas_scores = evaluate(
        ragas_dataset,
        metrics = [_answer_similarity],
        llm = judge_llm,
        embeddings = judge_embeddings,
        run_config = RunConfig(
            timeout = 120,
            max_retries = 10,      
            max_wait = 60,
            max_workers = 2
        )
    )

    
    # averaging 
    detector_time /= len(eval_data)
    detector_acc /= len(eval_data)

    extractor_time /= n_jobs_has_tools
    extractor_pre /= n_jobs_has_tools
    extractor_rec /= n_jobs_has_tools
    extractor_f1 /= n_jobs_has_tools

    enhancer_time /= len(eval_data)
    answer_similarity = ragas_scores["answer_similarity"]
    valid_sims = [s for s in answer_similarity if not pd.isna(s)]
    avg_similarity = sum(valid_sims) / len(valid_sims) if valid_sims else 0

        
    return {
        'avg_detector_time' : detector_time,
        'avg_detector_acc'  : detector_acc,
        'avg_extractor_time': extractor_time,
        'avg_extractor_pre' : extractor_pre,
        'avg_extractor_rec' : extractor_rec,
        'avg_extractor_f1'  : extractor_f1,
        "enhancer_time"     : enhancer_time,
        "answer_similarity" : avg_similarity,
    }


def construct_ragas_question(client, job_enhnacer: str, job_desc: str, extracted_tools: list, **kwargs):
    if len(extracted_tools):
        tools_str = ", ".join(extracted_tools)
        
        system_prompt = get_enhancement_prompt(True)

        question = f"""Enhance this job description to be more professional:
{job_desc}


Use these tools in your enhanced response:
Tools: {tools_str}
"""
    else:
        system_prompt = get_enhancement_prompt(False)
        question = f"""Enhance this job description to be more professional:
{job_desc}
"""
    
    model_output, inference_time = query_model(
        client = client,
        query = job_desc,
        model_name = job_enhnacer,
        system_prompt = system_prompt,
        max_tokens = 1024,
        **kwargs
    )

    return question, model_output, inference_time





def get_enhancement_prompt(use_rag: bool) -> str:
    """
    Construct the System Prompt required for enhancing the job description.
    """

    if use_rag:
        prompt = """
You are an expert job poster on a freelancing platform.

Your role is to professionally enhance, structure, and rewrite the provided job description
in order to clarify the project scope, expectations, and requirements.

You will also be provided with additional tools/frameworks retrieved from external sources.
Integrate these tools into the enhanced description naturally and professionally where appropriate.

Your response MUST strictly follow the structure below:

## Overview
- Provide a clear and concise enhanced overview of the project.

## Requirements
- List the required responsibilities & tasks needed to complete the project.

## Tools / Frameworks Required
- List the relevant tools, technologies, frameworks, or platforms required for the project.

## Deliverables
- List the output files, visualizations, data, etc...
"""

    else:
        prompt = """
You are an expert job poster on a freelancing platform.

Your role is to professionally enhance, structure, and rewrite the provided job description
in order to clarify the project scope, expectations, and requirements.

Your response MUST strictly follow the structure below:

## Overview
- Provide a clear and concise enhanced overview of the project.

## Requirements
- List the required responsibilities & tasks needed to complete the project.

## Tools / Frameworks Required
- List the relevant tools, technologies, frameworks, or platforms that mentioned in the given description.

## Deliverables
- List the output files, visualizations, data, etc...
"""


    return prompt.strip()
