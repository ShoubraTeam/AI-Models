import os
import re
import json
from time import time
from pathlib import Path
from pprint import pprint
import logging
import traceback

import helpers.config as CFG
import helpers.functional as F 

from .evaluation_classes import EvaluationDataParser, AgentsInitializer, AgentsEvaluator
from agents import NumericalAnalyzer, BioAnalyzer, SkillsAnalyzer, VisualBrandEvaluator, SuperAgent
from .handle_errors import get_short_error_info

Agent_Type = NumericalAnalyzer | \
    BioAnalyzer      | \
    SkillsAnalyzer      | \
    VisualBrandEvaluator  | \
    SuperAgent   

# -------------------------------------------- SuperAgent Evaluation Wrapper --------------------------------------------

class SuperAgentEvaluationWrapper:
    """
    A Proxy/Adapter Wrapper that conforms strictly to the single-argument execution loop
    required by AgentsEvaluator, while transparently orchestrating the 4 sub-agents dependencies.
    """
    def __init__(self, super_agent, model_name: str, **kwargs):
        self.super_agent = super_agent
        
        visual_model_resolved = CFG.EVALUATION_MODELS_MAPPING.get("GEMINI_FLASH")
        skills_model_resolved = CFG.EVALUATION_MODELS_MAPPING.get("LLAMA_8B")
        numerical_model_resolved = CFG.EVALUATION_MODELS_MAPPING.get("LLAMA_8B")
        
        self.visual_agent = get_agent(CFG.TASK_VISUAL_BRAND_ANALYSIS, visual_model_resolved, **kwargs)
        self.bio_agent = get_agent(CFG.TASK_BIO_ANALYSIS, model_name, **kwargs)
        self.skills_agent = get_agent(CFG.TASK_SKILLS_ANALYSIS, skills_model_resolved, **kwargs)
        self.numerical_agent = get_agent(CFG.TASK_NUMERICAL_ANALYSIS, numerical_model_resolved, **kwargs)

    def evaluate_sample(self, sample: dict) -> dict:
        """Intercepts the flat sample loop, computes sub-audits, and feeds the Master Orchestrator."""
        base_dir = Path(__file__).resolve().parents[2]
        img_path = os.path.join(base_dir, sample.get("image_name", ""))
        if not os.path.exists(img_path):
            img_path = os.path.join(base_dir, "app", sample.get("image_name", ""))
        if not os.path.exists(img_path):
            img_path = os.path.join(os.getcwd(), sample.get("image_name", ""))
            
        visual_res = self.visual_agent.invoke(image_path=img_path, job_role=sample.get("job_role", ""))
        bio_res = self.bio_agent.invoke(bio_text=sample.get("bio_text", ""), job_role=sample.get("job_role", ""))
        skills_res = self.skills_agent.invoke(declared_skills=sample.get("declared_skills", []), job_role=sample.get("job_role", ""))
        numerical_res = self.numerical_agent.invoke(
            job_role=sample.get("job_role", ""),
            hourly_rate=sample.get("hourly_rate", 0.0),
            rating=sample.get("rating", 0.0),
            total_completed_jobs=sample.get("total_completed_jobs", 0)
        )
        
        return self.super_agent.evaluate_sample(
            sample=sample,
            visual_res=visual_res,
            bio_res=bio_res,
            skills_res=skills_res,
            numerical_res=numerical_res
        )

    def __getattr__(self, name):
        """Delegates all standard attributes and metadata methods (like get_metric_names) to the inner agent."""
        return getattr(self.super_agent, name)


# -------------------------------------------- Acquiring Agents & Data --------------------------------------------------

def get_task_eval_data(all_data: list[dict], task_name: str) -> list[dict]:
    """
    Parse the given whole data and returns the task specific evaluation data,
    while explicitly merging back the ground-truth logs and metrics context.
    """
    task_data = []

    for sample in all_data:
        if task_name == CFG.TASK_NUMERICAL_ANALYSIS:
            parsed = EvaluationDataParser.get_numerical_data(sample = sample)
        elif task_name == CFG.TASK_BIO_ANALYSIS:
            parsed = EvaluationDataParser.get_bio_data(sample = sample)
        elif task_name == CFG.TASK_SKILLS_ANALYSIS:
            parsed = EvaluationDataParser.get_skills_data(sample = sample)
        elif task_name == CFG.TASK_VISUAL_BRAND_ANALYSIS:
            parsed = EvaluationDataParser.get_visual_data(sample = sample)
        else:
            parsed = EvaluationDataParser.get_super_agent_data(sample = sample)

        if isinstance(parsed, dict):
            base_dir = Path(__file__).resolve().parents[2]
            abs_img_path = os.path.join(base_dir, sample.get("image_name", ""))
            if not os.path.exists(abs_img_path):
                abs_img_path = os.path.join(base_dir, "app", sample.get("image_name", ""))
            if not os.path.exists(abs_img_path):
                abs_img_path = os.path.join(os.getcwd(), sample.get("image_name", ""))
                
            parsed["image_path"] = abs_img_path
            parsed["ground_truth_sub_audits"] = sample.get("ground_truth_sub_audits", {})
            parsed["ground_truth_orchestrator"] = sample.get("ground_truth_orchestrator", {})
            parsed["freelancer_name"] = sample.get("freelancer_name", "")
            parsed["bio_text"] = sample.get("bio_text", "")
            parsed["image_name"] = sample.get("image_name", "")
            parsed["declared_skills"] = sample.get("declared_skills", [])
            parsed["hourly_rate"] = sample.get("hourly_rate", 0.0)
            parsed["rating"] = sample.get("rating", 0.0)
            parsed["total_completed_jobs"] = sample.get("total_completed_jobs", 0)
            parsed["job_role"] = sample.get("job_role", parsed.get("job_role", ""))

        task_data.append(parsed)

    return task_data


def get_eval_data(task_name: str) -> list[dict]:
    """Load the data & parse the task specific data from it."""
    data = F.load_json(CFG.EVAL_DATA_PATH)
    task_data = get_task_eval_data(all_data = data, task_name = task_name)
    return task_data


def get_agent(
    task_name : str,
    model_name: str,
    **kwargs
):
    """Acquiring the agent for the given task"""
    if task_name == CFG.TASK_NUMERICAL_ANALYSIS:
        return AgentsInitializer.get_numerical_analyzer_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_BIO_ANALYSIS:
        return AgentsInitializer.get_bio_analyzer_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_SKILLS_ANALYSIS:
        return AgentsInitializer.get_skills_analyzer_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_VISUAL_BRAND_ANALYSIS:
        return AgentsInitializer.get_visual_brand_evaluator_agent(model_name = model_name, **kwargs)
   
    if task_name == CFG.TASK_SUPER_AGENT:
        return AgentsInitializer.get_super_agent(model_name = model_name, **kwargs)
    
    raise ValueError("Task is not allowed")


# -------------------------------------------- Evaluation --------------------------------------------------

def evaluate_agent_on_task(
    run_id              : int,
    task_name           : str,
    model_name          : str,
    **kwargs            
) -> None:
    """General Function for evaluating agents"""

    if task_name not in CFG.ALLOWED_EVALUATION_TASKS:
        raise ValueError("Task is not allowed")
    
    try:
        agent = get_agent(
            task_name  = task_name,
            model_name = model_name,
            **kwargs
        )
        
        if task_name == CFG.TASK_SUPER_AGENT:
            agent = SuperAgentEvaluationWrapper(super_agent=agent, model_name=model_name, **kwargs)
            
        F.print_success_message(f">> Agent Loaded Successfully: [Task: {task_name} | Model: {model_name}].")
    
    except Exception as e:
        F.print_error_message(f">> Error While Loading Agent: [Task: {task_name} | Model: {model_name}].")
        F.print_error_message(">> Error Info:")
        for key, val in get_short_error_info(e).items():
            print(f"\t- {key} --> {val}")
        raise
    
    try:
        eval_data = get_eval_data(task_name = task_name)
        F.print_success_message(f">> Data Loaded Successfully: [Task: {task_name} | Model: {model_name}].")
    except Exception as e:
        F.print_error_message(f">> Error While Loading Data: [Task: {task_name} | Model: {model_name}].")
        F.print_error_message(">> Error Info:")
        for key, val in get_short_error_info(e).items():
            print(f"\t- {key} --> {val}")
        raise

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
        F.print_error_message(">> FULL CRASH TRACEBACK INFO:")
        traceback.print_exc()
        raise e