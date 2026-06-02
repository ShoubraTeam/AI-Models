# ------------------------------------------------------
# Agents Evaluation
# ------------------------------------------------------

import helpers.config as CFG
import helpers.functional as F 

from .evaluation_classes import EvaluationDataParser, AgentsInitializer, AgentsEvaluator

from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator
from agents import JobRequirementsExtractor, JobRequirementsMatcher
from agents import ExperienceEvidenceAgent
from agents import LanguageClarityEvaluator
from .handle_errors import get_short_error_info

Agent_Type = JobToolsExtractor | \
    ProposalToolsAnalyzer      | \
    JobKeyPointsExtractor      | \
    JobUnderstandingEvaluator  | \
    JobRequirementsExtractor   | \
    JobRequirementsMatcher     | \
    ExperienceEvidenceAgent    | \
    LanguageClarityEvaluator
# -------------------------------------------- Acquiring Agents & Data --------------------------------------------------
def get_task_eval_data(all_data: list[dict], task_name: str) -> list[dict]:
    """
    Parse the given whole data and returns the task specific evaluation data

    Args:
        all_data      : the list of all data objects
        task_name     : the name of the task

    Returns:
        task_data (list): the list of the specific task data
    """
    task_data = []

    for sample in all_data:
        if task_name == CFG.TASK_JOB_TOOLS_EXTRACTOR:
            task_data.append(EvaluationDataParser.get_job_tools_extractor_data(sample = sample))

        elif task_name == CFG.TASK_PROPOSAL_TOOLS_ANALYZER:
            task_data.append(EvaluationDataParser.get_proposal_tools_analyzer_data(sample = sample))

        elif task_name == CFG.TASK_JOB_REQUIREMENTS_EXTRACTOR:
            task_data.append(EvaluationDataParser.get_job_requirements_extractor_data(sample = sample))
            
        elif task_name == CFG.TASK_JOB_REQUIREMENTS_MATCHER:
            task_data.append(EvaluationDataParser.get_job_requirements_matcher_data(sample = sample))

        elif task_name == CFG.TASK_JOB_KEY_POINTS_EXTRACTOR:
            task_data.append(EvaluationDataParser.get_job_key_points_extractor_data(sample = sample))

        elif task_name == CFG.TASK_JOB_UNDERSTANDING_EVALUATOR:
            task_data.append(EvaluationDataParser.get_job_understanding_evaluator_data(sample = sample))
        
        elif task_name == CFG.TASK_EXPERIENCE_EVIDENCE_FINDER:
            task_data.append(EvaluationDataParser.get_experience_evidence_finder_data(sample = sample))

        elif task_name == CFG.TASK_LANGUAGE_CLARITY_EVALUATOR:
            task_data.append(EvaluationDataParser.get_language_clarity_evaluator_data(sample = sample))
        
        else:
            task_data.append(EvaluationDataParser.get_super_agent_data(sample = sample))

    return task_data



def get_eval_data(task_name: str) -> list[dict]:
    """
    Load the data & parse the task specific data from it.

    Args:
        data_file_name: the stored JSON filename
        task_name     : the name of the task
    """
    data = F.load_json(CFG.EVAL_DATA_PATH)

    task_data = get_task_eval_data(all_data = data, task_name = task_name)
    return task_data


def get_agent(
    task_name : str,
    model_name: str,
    **kwargs
):
    """Acquiring the agent for the given task"""
    if task_name == CFG.TASK_JOB_TOOLS_EXTRACTOR:
        return AgentsInitializer.get_job_tool_extractor_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_PROPOSAL_TOOLS_ANALYZER:
        return AgentsInitializer.get_proposal_tool_analyzer_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_JOB_REQUIREMENTS_EXTRACTOR:
        return AgentsInitializer.get_job_requirements_extractor_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_JOB_REQUIREMENTS_MATCHER:
        return AgentsInitializer.get_job_requirements_matcher_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_JOB_KEY_POINTS_EXTRACTOR:
        return AgentsInitializer.get_job_key_points_extractor_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_JOB_UNDERSTANDING_EVALUATOR:
        return AgentsInitializer.get_job_understanding_evaluator_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_EXPERIENCE_EVIDENCE_FINDER:
        return AgentsInitializer.get_experience_evidence_finder_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_LANGUAGE_CLARITY_EVALUATOR:
        return AgentsInitializer.get_language_clarity_evaluator_agent(model_name = model_name, **kwargs)
   


# -------------------------------------------- Evaluation --------------------------------------------------

def evaluate_agent_on_task(
    run_id              : int,
    task_name           : str,
    model_name          : str,
    **kwargs            
) -> None:
    """
    General Function for evaluating agents

    Args:
        run_id    (int): the count of this evaluation experiment 
        task_name (str): string determines the agents to evaluate:
            - job_tools_extractor
            - proposal_tools_analyzer
            - job_requirements_extractor
            - job_requirements_matcher
            - language_clarity_evaluator
            - job_key_points_extractor
            - job_understanding_evaluator
            - experience_evidence_finder
        
        model_name          (str): list of models names to evaluate    
        system_prompt       (str): the system prompts for model_1, model_2, ...
        structured_response (str): the structured responses for model_1, model_2, ...
        **kwargs           (dict): model kw-args [temperature - max_tokens - top_p]
    
    Returns:
        average_scores: the average of scores for the agent
    """

    if task_name not in CFG.ALLOWED_EVALUATION_TASKS:
        raise ValueError("Task is not allowed")
    
    # get agents & eval data
    try:
        agent = get_agent(
        task_name  = task_name,
        model_name = model_name,
        **kwargs
    )
        F.print_success_message(f">> Agent Loaded Successfuly: [Task: {task_name} | Model: {model_name}].")
    
    except Exception as e:
        F.print_error_message(f">> Error While Loading Agent: [Task: {task_name} | Model: {model_name}].")
        F.print_error_message(">> Error Info:")
        for key, val in get_short_error_info(e).items():
            print(f"\t- {key} --> {val}")
        raise
    
    
    try:
        eval_data = get_eval_data(task_name = task_name)
        F.print_success_message(f">> Data Loaded Successfuly: [Task: {task_name} | Model: {model_name}].")
    except Exception as e:
        F.print_error_message(f">> Error While Loading Data: [Task: {task_name} | Model: {model_name}].")
        F.print_error_message(">> Error Info:")
        for key, val in get_short_error_info(e).items():
            print(f"\t- {key} --> {val}")
        raise


    # evaluate
    run_configurations = {
        "run_id"      : run_id,
        "model_name"  : model_name,
        "model_kwargs": kwargs
    }

    evaluator = AgentsEvaluator(
        task_name          = task_name,
        agent              = agent,
        data               = eval_data,
        run_configurations = run_configurations,
        reset_results      = run_id == 1
    )

    try:
        evaluator.evaluate()
        F.print_success_message(f">> Agent Evaluated Successfully: [Task: {task_name} | Model: {model_name}].")
    except Exception as e:
        F.print_error_message(f">> Error while Evaluating: [Task: {task_name} | Model: {model_name}].")
        F.print_error_message(">> Error Info:")
        for key, val in get_short_error_info(e).items():
            print(f"\t- {key} --> {val}")

    
