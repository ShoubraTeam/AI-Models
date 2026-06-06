import re
import json
from time import time
from agents.BaseAgent import BaseAgent
from schemas import SuperAgentSchema
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

class SuperAgent(BaseAgent):
    """
    The Master Orchestrator Agent. Consumes outputs from all other sub-agents,
    performs cross-domain reasoning, and compiles the final Executive Audit Report.
    Includes built-in Hybrid Evaluation Engine to audit cross-domain synthesis quality.
    """
    def __init__(self, model_name: str, system_prompt: str, tools: list = [], structured_response = None, **kwargs):
        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)
        self.case_counter = 0

    def invoke(self, visual_res, bio_res, skills_res, numerical_res) -> SuperAgentSchema:
        
        def to_dict(obj):
            if hasattr(obj, "dict"): return obj.dict()
            if hasattr(obj, "model_dump"): return obj.model_dump()
            return obj

        formatted_input = (
            "=== SUB-AUDIT 1: VISUAL BRAND ===\n"
            f"{json.dumps(to_dict(visual_res), indent=2)}\n\n"
            "=== SUB-AUDIT 2: BIO COPYWRITING ===\n"
            f"{json.dumps(to_dict(bio_res), indent=2)}\n\n"
            "=== SUB-AUDIT 3: SKILLS ALIGNMENT ===\n"
            f"{json.dumps(to_dict(skills_res), indent=2)}\n\n"
            "=== SUB-AUDIT 4: NUMERICAL METRICS ENGINE ===\n"
            f"{json.dumps(to_dict(numerical_res), indent=2)}\n"
        )
        
        return super().invoke(input=formatted_input)

    def _get_semantic_orchestrator_coverage(self, expected_items: list[str], pred_items: list[str]) -> int:
        """
        LLM Judge: Evaluates semantic containment between the ground-truth core concepts 
        and the agent's synthesized text arrays.
        """
        if not expected_items or not pred_items:
            return 0

        judge_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)
        
        expected_str = "\n".join([f"- [{i}] {e}" for i, e in enumerate(expected_items)])
        pred_str = "\n".join([f"- {p}" for p in pred_items])
        
        prompt = f"""
        You are a senior executive auditor. Your task is to verify if the mandatory strategic audit points from the "Expected Points" list are successfully captured and covered inside the agent's "Predicted Synthesis" report list.
        
        Expected Points (Ground Truth Targets):
        {expected_str}
        
        Predicted Synthesis (Agent Output Report):
        {pred_str}
        
        Instructions:
        1. Analyze each item in the Expected list, and determine if its semantic core meaning is fully addressed, logged, or mentioned inside the Predicted list.
        2. Output ONLY the indices of the Expected list that were successfully covered, separated by commas (e.g., 0, 2). If none are covered, output "NONE".
        3. Strictly do NOT provide explanations, analysis notes, or markdown formatting. Output ONLY the raw list of indices or "NONE".
        """
        try:
            response = judge_model.invoke([HumanMessage(content=prompt)])
            cleaned_res = response.content.strip().upper()
            
            if "NONE" in cleaned_res or not cleaned_res:
                return 0
            
            matched_indices = re.findall(r'\d+', cleaned_res)
            unique_matches = {int(idx) for idx in matched_indices if int(idx) < len(expected_items)}
            return len(unique_matches)
            
        except Exception as e:
            print(f" -> [ORCHESTRATOR JUDGE ERROR] {e}")
            return 0

    def get_metric_names(self) -> tuple[str, str, str, str, str]:
        return (
            "orchestrator_score_mae",
            "orchestrator_score_in_range_accuracy",
            "weaknesses_coverage_ratio",
            "action_plan_coverage_ratio",
            "agent_invocation_time"
        )

    def evaluate_sample(self, sample: dict, visual_res, bio_res, skills_res, numerical_res) -> dict:
        """
        Evaluates the Master Orchestrator (SuperAgent) by passing pre-computed sub-audit objects.
        Validates the final weighted score mathematically, and report synthesis text semantically.
        """
        self.case_counter += 1 
        print(f"\n" + "="*40 + f" [EVALUATING SUPERAGENT CASE #{self.case_counter}] " + "="*40)

        gt_orch = sample.get("ground_truth_orchestrator", {})
        true_range = gt_orch.get("true_overall_score_range", {"min": 0.0, "max": 1.0})
        req_weaknesses = gt_orch.get("required_weaknesses_mentions", [])
        req_actions = gt_orch.get("required_action_items", [])

        start_time = time()
        agent_output = self.invoke(
            visual_res=visual_res, 
            bio_res=bio_res, 
            skills_res=skills_res, 
            numerical_res=numerical_res
        )
        end_time = time()
        invocation_time = end_time - start_time

        pred_score = agent_output.overall_score
        pred_weaknesses = agent_output.critical_weaknesses if agent_output.critical_weaknesses else []
        pred_actions = agent_output.prioritized_action_plan if agent_output.prioritized_action_plan else []

        print(f"[METRICS] True Range: [{true_range['min']} - {true_range['max']}] | Agent Overall Score: {pred_score}")
        print(f"[SYNTHESIS] Expected Weaknesses Mentions: {req_weaknesses}")
        print(f"[SYNTHESIS] Agent Predicted Weaknesses:  {pred_weaknesses}")

        midpoint = (true_range["min"] + true_range["max"]) / 2
        score_mae = abs(pred_score - midpoint)
        score_in_range = 1.0 if true_range["min"] <= pred_score <= true_range["max"] else 0.0

        matched_weaknesses = self._get_semantic_orchestrator_coverage(req_weaknesses, pred_weaknesses)
        weaknesses_ratio = matched_weaknesses / len(req_weaknesses) if req_weaknesses else 1.0

        matched_actions = self._get_semantic_orchestrator_coverage(req_actions, pred_actions)
        actions_ratio = matched_actions / len(req_actions) if req_actions else 1.0

        print(f"[EVAL RESULT] Weaknesses Mentions Caught: {matched_weaknesses}/{len(req_weaknesses)} | Action Items Caught: {matched_actions}/{len(req_actions)}")
        print("-" * 100)

        return {
            "orchestrator_score_mae": round(score_mae, 4),
            "orchestrator_score_in_range_accuracy": score_in_range,
            "weaknesses_coverage_ratio": round(weaknesses_ratio, 4),
            "action_plan_coverage_ratio": round(actions_ratio, 4),
            "agent_invocation_time": round(invocation_time, 4)
        }