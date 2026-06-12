import re
from time import time
from agents.profile_analysis.BaseAgent import BaseAgent
from models.schemas import BioAnalyzerSchema
from models.config.agents_config import PA_DEFAULT_MODELS_CFG
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

class BioAnalyzer(BaseAgent):

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        model_provider: str = "groq",
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = PA_DEFAULT_MODELS_CFG["bio_analyzer"]

        super().__init__(model_name, system_prompt, tools, structured_response, model_provider, **kwargs)
        self.case_counter = 0 

    def invoke(self, bio_text: str, job_role: str) -> BioAnalyzerSchema:
        formatted_input = (
            f"Freelancer Target Job Role: {job_role}\n"
            f"Freelancer Profile Bio/Summary Text:\n"
            f"\"\"\"\n{bio_text}\n\"\"\""
        )
        return super().invoke(input=formatted_input)

    def _get_semantic_content_coverage(self, expected_concerns: list[str], pred_bullets: list[str]) -> int:
        """
        LLM Judge: Compares the generated analysis bullets against the ground truth concerns.
        Returns the exact count of successfully identified flaws based on meaning.
        """
        if not expected_concerns or not pred_bullets:
            return 0

        judge_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)
        
        expected_str = "\n".join([f"- [{i}] {e}" for i, e in enumerate(expected_concerns)])
        pred_str = "\n".join([f"- {p}" for p in pred_bullets])
        
        prompt = f"""
        You are a quality control auditor. Your job is to check which profile flaws from the "Expected Concerns" list are successfully identified inside the agent's "Predicted Analysis" bullets.
        
        Expected Concerns (Ground Truth Flaws):
        {expected_str}
        
        Predicted Analysis (Agent Output Bullets):
        {pred_str}
        
        Instructions:
        1. Read each concern in the Expected list, and check if the Predicted list mentions or points out that exact flaw (even if using different wording).
        2. Output ONLY the indices of the Expected list that were successfully covered, separated by commas (e.g., 0, 2). If none are covered, output "NONE".
        3. Strictly do NOT write explanations, markdown syntax, or extra text. Output ONLY the raw list of indices or "NONE".
        """
        try:
            response = judge_model.invoke([HumanMessage(content=prompt)])
            cleaned_res = response.content.strip().upper()
            
            if "NONE" in cleaned_res or not cleaned_res:
                return 0
            
            matched_indices = re.findall(r'\d+', cleaned_res)
            unique_matches = {int(idx) for idx in matched_indices if int(idx) < len(expected_concerns)}
            return len(unique_matches)
            
        except Exception as e:
            print(f" -> [BIO JUDGE ERROR] {e}")
            return 0

    def get_metric_names(self) -> tuple[str, str, str, str]:
        """Returns the precise metric names for the final analytics tables."""
        return (
            "bio_score_mae",
            "bio_score_in_range_accuracy",
            "bio_content_coverage_ratio",
            "agent_invocation_time"
        )

    def evaluate_sample(self, sample: dict) -> dict:
        """
        Evaluates the Bio Agent output against ground truth.
        Scores are calculated mathematically, texts are evaluated semantically.
        """
        self.case_counter += 1 
        print(f"\n" + "="*40 + f" [EVALUATING BIO CASE #{self.case_counter}] " + "="*40)

        bio_text = sample.get("bio_text", "")
        job_role = sample.get("job_role", "")
        
        gt_bio = sample.get("ground_truth_sub_audits", {}).get("bio_analysis", {})
        true_range = gt_bio.get("true_score_range", {"min": 0.0, "max": 1.0})
        expected_cliches = gt_bio.get("expected_cliches", [])
        expected_dilutions = gt_bio.get("expected_dilutions", [])
        expected_concerns = expected_cliches + expected_dilutions 

        start_time = time()
        agent_output = self.invoke(bio_text=bio_text, job_role=job_role)
        end_time = time()
        invocation_time = end_time - start_time

        pred_score = agent_output.score
        pred_analysis = agent_output.analysis if agent_output.analysis else []

        print(f"[METRICS] True Range: [{true_range['min']} - {true_range['max']}] | Predicted: {pred_score}")
        print(f"[CONTENT] Expected Flaws: {expected_concerns}")
        print(f"[CONTENT] Agent Output: {pred_analysis}")

        midpoint = (true_range["min"] + true_range["max"]) / 2
        score_mae = abs(pred_score - midpoint)
        score_in_range = 1.0 if true_range["min"] <= pred_score <= true_range["max"] else 0.0

        matched_count = self._get_semantic_content_coverage(expected_concerns, pred_analysis)
        coverage_ratio = matched_count / len(expected_concerns) if expected_concerns else 1.0
        
        print(f"[EVAL RESULT] LLM Judge matched {matched_count}/{len(expected_concerns)} true concerns.")
        print("-" * 100)

        return {
            "bio_score_mae": round(score_mae, 4),
            "bio_score_in_range_accuracy": score_in_range,
            "bio_content_coverage_ratio": round(coverage_ratio, 4),
            "agent_invocation_time": round(invocation_time, 4)
        }