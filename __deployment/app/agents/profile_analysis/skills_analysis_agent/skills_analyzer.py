import re
from time import time
from typing import List
from agents.profile_analysis.BaseAgent import BaseAgent
from models import SkillsAnalyzerSchema  
from helpers.config import DEFAULT_MODELS_CFG


class SkillsAnalyzer(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG.get("skills_analyzer", {"temperature": 0.1})

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)
        self.case_counter = 0

    def invoke(
        self,
        declared_skills: List[str],
        job_role: str
    ) -> SkillsAnalyzerSchema:
        
        skills_string = ", ".join(declared_skills)
        
        formatted_input = (
            f"Target Job Role to Match: {job_role}\n"
            f"Freelancer Declared Skills List: [{skills_string}]"
        )
        return super().invoke(input=formatted_input)

    def _normalize_skill_name(self, skill: str) -> str:
        """
        Cleans and normalizes skill names to ensure robust matching.
        Transforms 'TensorFlow' or 'tensor-flow' into 'tensorflow'.
        """
        return re.sub(r'[^a-z0-9]', '', skill.lower().strip())

    def _calculate_list_metrics(self, true_list: list[str], pred_list: list[str]) -> tuple[float, float]:
        """
        Computes exact Precision and Recall using pure Python set operations 
        after string normalization. No LLM required.
        """
        if not true_list and not pred_list:
            return 1.0, 1.0  
        if not true_list and pred_list:
            return 0.0, 1.0  
            
        true_set = {self._normalize_skill_name(s) for s in true_list if s}
        pred_set = {self._normalize_skill_name(s) for s in pred_list if s}
        
        intersection = true_set.intersection(pred_set)
        
        recall = len(intersection) / len(true_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        
        return precision, recall

    def get_metric_names(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            "skills_score_mae",
            "skills_score_in_range_accuracy",
            "missing_skills_precision",
            "missing_skills_recall",
            "irrelevant_skills_precision",
            "irrelevant_skills_recall",
            "agent_invocation_time"
        )

    def evaluate_sample(self, sample: dict) -> dict:
        """
        Evaluates the Technical Skills Agent output against Ground Truth.
        Compares explicit technology keywords instantly using pure Python logic.
        """
        self.case_counter += 1 
        print(f"\n" + "="*40 + f" [EVALUATING SKILLS CASE #{self.case_counter}] " + "="*40)

        declared_skills = sample.get("declared_skills", [])
        job_role = sample.get("job_role", "")
        
        gt_skills = sample.get("ground_truth_sub_audits", {}).get("skills_alignment", {})
        true_range = gt_skills.get("true_score_range", {"min": 0.0, "max": 1.0})
        true_missing = gt_skills.get("true_missing_essential", [])
        true_irrelevant = gt_skills.get("true_irrelevant_skills", [])

        start_time = time()
        agent_output = self.invoke(declared_skills=declared_skills, job_role=job_role)
        end_time = time()
        invocation_time = end_time - start_time

        pred_score = agent_output.score
        pred_missing = agent_output.missing_essential_skills if agent_output.missing_essential_skills else []
        pred_irrelevant = agent_output.irrelevant_skills if agent_output.irrelevant_skills else []

        print(f"[METRICS] True Range: [{true_range['min']} - {true_range['max']}] | Agent Predicted Score: {pred_score}")
        print(f"[MISSING TECH] True Expected: {true_missing} | Agent Output: {pred_missing}")
        print(f"[IRRELEVANT]   True Expected: {true_irrelevant} | Agent Output: {pred_irrelevant}")

        midpoint = (true_range["min"] + true_range["max"]) / 2
        score_mae = abs(pred_score - midpoint)
        score_in_range = 1.0 if true_range["min"] <= pred_score <= true_range["max"] else 0.0

        missing_prec, missing_rec = self._calculate_list_metrics(true_missing, pred_missing)

        irrelevant_prec, irrelevant_rec = self._calculate_list_metrics(true_irrelevant, pred_irrelevant)

        print(f"[EVAL RESULT] Missing Recall: {round(missing_rec, 2)} | Irrelevant Recall: {round(irrelevant_rec, 2)}")
        print("-" * 100)

        return {
            "skills_score_mae": round(score_mae, 4),
            "skills_score_in_range_accuracy": score_in_range,
            "missing_skills_precision": round(missing_prec, 4),
            "missing_skills_recall": round(missing_rec, 4),
            "irrelevant_skills_precision": round(irrelevant_prec, 4),
            "irrelevant_skills_recall": round(irrelevant_rec, 4),
            "agent_invocation_time": round(invocation_time, 4)
        }