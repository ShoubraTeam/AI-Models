# ------------------------------------------------------
# Agents Evaluation
# ------------------------------------------------------


from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobRequirementsExtractor, JobRequirementsMatcher
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator
import helpers.config as CFG
import helpers.functional as F 
from collections import defaultdict
import os


def get_eval_data(data_file_name: str):
    file_path = os.path.join(CFG.EVAL_DATA_PATH, data_file_name)
    data = F.load_json(file_path)
    return data


def get_agents(
    task_name: str,
    models: list[str],
    system_prompts: list[str],
    structured_responses: list,
    **kwargs
):
    """Acquiring the agents with thier functionality names"""
    if task_name == CFG.TOOLS_ALIGNMENT_TASK:
        job_tools_extractor = JobToolsExtractor(
            model_name = models[0],
            system_prompt = system_prompts[0],
            structured_response = structured_responses[0],
            **kwargs
        )

        proposal_tools_analyzer = ProposalToolsAnalyzer(
            model_name = models[1],
            system_prompt = system_prompts[1],
            structured_response = structured_responses[1],
            **kwargs
        )

        agents = [
            ("job_tools_extractor"     , job_tools_extractor),
            ("proposal_tools_analyzer" , proposal_tools_analyzer)
        ]

        return agents


    if task_name == CFG.JOB_UNDERSTANDING_TASK:
        job_key_points_extractor = JobKeyPointsExtractor(
            model_name = models[0],
            system_prompt = system_prompts[0],
            structured_response = structured_responses[0],
            **kwargs
        )

        job_understanding_evaluator = JobUnderstandingEvaluator(
            model_name = models[1],
            system_prompt = system_prompts[1],
            structured_response = structured_responses[1],
            **kwargs
        )

        agents = [
            ("job_key_points_extractor"    , job_key_points_extractor),
            ("job_understanding_evaluator" , job_understanding_evaluator)
        ]

        return agents

    if task_name == CFG.REQUIREMENT_COVERAGE_TASK:
        job_requirements_extractor = JobRequirementsExtractor(
            model_name = models[0],
            system_prompt = system_prompts[0],
            structured_response = structured_responses[0],
            **kwargs
        )

        job_requirements_matcher = JobRequirementsMatcher(
            model_name = models[1],
            system_prompt = system_prompts[1],
            structured_response = structured_responses[1],
            **kwargs
        )

        agents = [
            ("job_requirements_extractor", job_requirements_extractor),
            ("job_requirements_matcher"  , job_requirements_matcher)
        ]

        return agents

    if task_name == CFG.EVIDENCE_OF_EXPERIENCE_TASK:
        pass

    if task_name == CFG.LANGUAGE_CLARITY_TASK:
        pass

    if task_name == CFG.SUPER_AGENT_TASK:
        pass


def calc_avg_for_multiple_metrics(scores: list[dict]):
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
    task_name: str,
    models: list[str],
    system_prompts: list[str],
    structured_responses: list,
    eval_data_file_name: str,
    rounds: int = 5,
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
    eval_data = get_eval_data(data_file_name = eval_data_file_name)


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