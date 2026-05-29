# ------------------------------------------------------
# Agents Evaluation
# ------------------------------------------------------





import helpers.config as CFG
import helpers.functional as F 
from collections import defaultdict
import os

from .task_data_parser import EvaluationDataParser
from .agents_initializer import AgentsInitializer


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
        if task_name == CFG.TOOLS_ALIGNMENT_TASK:
            task_data.append(EvaluationDataParser.get_tools_alignment_data(sample = sample))

        elif task_name == CFG.REQUIREMENT_COVERAGE_TASK:
            task_data.append(EvaluationDataParser.get_requirement_coverage_data(sample = sample))

        elif task_name == CFG.JOB_UNDERSTANDING_TASK:
            task_data.append(EvaluationDataParser.get_job_understanding_data(sample = sample))
            
        elif task_name == CFG.EVIDENCE_OF_EXPERIENCE_TASK:
            task_data.append(EvaluationDataParser.get_evidence_of_experience_data(sample = sample))

        elif task_name == CFG.LANGUAGE_CLARITY_TASK:
            task_data.append(EvaluationDataParser.get_language_clarity_data(sample = sample))

        else:
            task_data.append(EvaluationDataParser.get_super_agent_data(sample = sample))

    return task_data


def get_eval_data(data_file_name: str, task_name: str) -> list[dict]:
    """
    Load the data & parse the task specific data from it.

    Args:
        data_file_name: the stored JSON filename
        task_name     : the name of the task
    """
    file_path = os.path.join(CFG.EVAL_DATA_PATH, data_file_name)
    data = F.load_json(file_path)

    task_data = get_task_eval_data(all_data = data, task_name = task_name)
    return task_data


def get_agents(
    task_name: str,
    models: list[str],
    system_prompts: list[str],
    structured_responses: list,
    **kwargs
):
    """Acquiring the agents with thier functional names"""
    if task_name == CFG.TOOLS_ALIGNMENT_TASK:
        return AgentsInitializer.get_tools_alignment_agents(
            models = models,
            system_prompts = system_prompts,
            structured_responses = structured_responses,
            **kwargs
        )

    if task_name == CFG.JOB_UNDERSTANDING_TASK:
        return AgentsInitializer.get_job_understanding_agents(
            models = models,
            system_prompts = system_prompts,
            structured_responses = structured_responses,
            **kwargs
        )
        

    if task_name == CFG.REQUIREMENT_COVERAGE_TASK:
        return AgentsInitializer.get_requirement_coverage_agents(
            models = models,
            system_prompts = system_prompts,
            structured_responses = structured_responses,
            **kwargs
        )
        

    if task_name == CFG.EVIDENCE_OF_EXPERIENCE_TASK:
        return AgentsInitializer.get_evidence_of_experience_agents(
            models = models,
            system_prompts = system_prompts,
            structured_responses = structured_responses,
            **kwargs
        )

    if task_name == CFG.LANGUAGE_CLARITY_TASK:
        return AgentsInitializer.get_language_clarity_agents(
            models = models,
            system_prompts = system_prompts,
            structured_responses = structured_responses,
            **kwargs
        )

    if task_name == CFG.SUPER_AGENT_TASK:
        return AgentsInitializer.get_super_agent(
            models = models,
            system_prompts = system_prompts,
            structured_responses = structured_responses,
            **kwargs
        )


# -------------------------------------------- Evaluation --------------------------------------------------

def calc_avg_for_multiple_metrics(scores: list[dict[str, float]]) -> defaultdict[str, float]:
    """
    Calculating the average for multiple metrics

    Args:
        scores: the list of given metrics

    Returns:
        averages: a dict contains mapping each metric to its corresponding average
    """
    if not isinstance(scores[0], dict):
        raise TypeError("The metrics should be a dict object")
    
    totals = defaultdict(float)

    for score_dict in scores:
        
        # calc total for each metric
        for key, val in score_dict.items():
            totals[key] += val
    
    averages = {
        key : total / len(scores)
        for key, total in totals.items()
    }

    return averages



def evaluate_agents_on_task(
    task_name           : str,
    models              : list[str],
    system_prompts      : list[str],
    structured_responses: list,
    eval_data_file_name : str,
    rounds              : int = 5,
    **kwargs
) -> dict:
    """
    General Function for evaluating agents

    Args:
        task_name (str): string determines the agents to evaluate:
            - tools_alignment
            - job_understanding
            - requirement_coverage
            - evidence_of_experience
            - language_clarity
            - super_agent
        
            
        models               (list): list of models names to evaluate    
        system_prompts       (list): the system prompts for model_1, model_2, ...
        structured_responses (list): the structured responses for model_1, model_2, ...
        eval_data_file_name  (str) : the eval data
        rounds               (int) : how many times to run the evaluation function (to average scores)
        kwargs               (dict): Kwargs should control the model behavior
    
    Returns:
        average_scores: the average of scores returned by the agents
            {
                "agent1" : scores,
                "agent2" : scores,
                ...
            }
    """

    if task_name not in CFG.ALLOWED_EVALUATION_TASKS:
        raise ValueError("Task is not allowed")
    

    # get agents to evaluate
    agents = get_agents(
        task_name = task_name,
        models = models,
        system_prompts = system_prompts,
        structured_responses = structured_responses,
        **kwargs
    )

    # eval data
    eval_data = get_eval_data(data_file_name = eval_data_file_name, task_name = task_name)
    F.print_eval_data(eval_data)

    # scoring
    average_scores = []
    for agent in agents:
        # get agent data
        agent_name = agent[0]
        agent_obj  = agent[1]
        F.print_subtitle(f"Evaluating: {agent_name}")

        # evaluate
        agent_scores = []
        for round in range(rounds):
            print(f">> Round {round + 1}")
            agent_scores.append(agent_obj.evaluate(eval_data))

        # calc avg
        if isinstance(agent_scores[0], dict):  # if multiple metrics
            agent_avg_scores = calc_avg_for_multiple_metrics(agent_scores)
        
        else: # single metric
            agent_avg_scores = sum(agent_scores) / len(agent_scores)

        average_scores.append({
            agent_name : agent_avg_scores
        })
    
    return average_scores