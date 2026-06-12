import os
import base64
import mimetypes
import re
from time import time
from pathlib import Path
import traceback
from agents.profile_analysis.BaseAgent import BaseAgent
from models.schemas import VisualBrandEvaluationSchema
from models.config.agents_config import PA_DEFAULT_MODELS_CFG
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


class VisualBrandEvaluator(BaseAgent):

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        model_provider: str = "google_genai",
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = PA_DEFAULT_MODELS_CFG["visual_brand_evaluator"]

        super().__init__(model_name, system_prompt, tools, structured_response, model_provider, **kwargs)
        self.case_counter = 0

    def _encode_image_to_base64(self, image_path: str) -> tuple:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"  
            
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
        return encoded_string, mime_type

    def invoke(self, image_path: str, job_role: str) -> VisualBrandEvaluationSchema:
        base64_image, mime_type = self._encode_image_to_base64(image_path)
        
        # الـ Structure الصح الصريح اللي موديل Gemini بيفهمه جوه LangChain
        multimodal_content = [
            {
                "type": "text", 
                "text": (
                    f"Freelancer Job Role: {job_role}\n\n"
                    f"Please analyze this freelancer profile image directly and strictly according to your system prompt instructions, "
                    f"evaluating its appropriateness specifically for a {job_role}."
                )
            },
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{base64_image}" # جوجل بيفهمها كـ string مباشر هنا جوه الحقل ده في النسخ الجديدة
            }
        ]
        
        # أو الصيغة الأضمن لـ Gemini لو لسه معترض:
        # multimodal_content = [
        #     {"type": "text", "text": f"Evaluate this image for a {job_role}"},
        #     {"type": "image_data", "image_data": {"data": base64_image, "mime_type": mime_type}}
        # ]
        
        return super().invoke(input=multimodal_content)
    
    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

    def _get_semantic_content_coverage(self, expected_flaws: list[str], pred_bullets: list[str]) -> int:
        """
        LLM Judge: Evaluates if the agent's visual feedback successfully caught 
        the expected ground-truth flaws semantically.
        """
        if not expected_flaws or not pred_bullets:
            return 0

        judge_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)
        
        expected_str = "\n".join([f"- [{i}] {e}" for i, e in enumerate(expected_flaws)])
        pred_str = "\n".join([f"- {p}" for p in pred_bullets])
        
        prompt = f"""
        You are a quality control auditor for profile image branding. Your task is to check which expected flaws from the "Expected Visual Flaws" list are successfully identified inside the agent's "Predicted Feedback" bullets.
        
        Expected Visual Flaws (Ground Truth):
        {expected_str}
        
        Predicted Feedback (Agent Visual Output):
        {pred_str}
        
        Instructions:
        1. Review each flaw in the Expected list and verify if the Predicted feedback points it out or mentions it (even if using different creative phrasing).
        2. Output ONLY the indices of the Expected list that were successfully covered, separated by commas (e.g., 0, 1). If none are covered, output "NONE".
        3. Do NOT provide markdown styling, reasoning, or introductions. Output ONLY the raw list of indices or "NONE".
        """
        try:
            response = judge_model.invoke([HumanMessage(content=prompt)])
            cleaned_res = response.content.strip().upper()
            
            if "NONE" in cleaned_res or not cleaned_res:
                return 0
            
            matched_indices = re.findall(r'\d+', cleaned_res)
            unique_matches = {int(idx) for idx in matched_indices if int(idx) < len(expected_flaws)}
            return len(unique_matches)
            
        except Exception as e:
            print(f" -> [VISUAL JUDGE ERROR] {e}")
            return 0

    def get_metric_names(self) -> tuple[str, str, str, str]:
        return (
            "visual_brand_score_mae",
            "visual_brand_score_in_range_accuracy",
            "visual_brand_content_coverage_ratio",
            "agent_invocation_time"
        )

    def evaluate_sample(self, sample: dict) -> dict:
        self.case_counter += 1 
        print(f"\n" + "="*40 + f" [EVALUATING VISUAL CASE #{self.case_counter}] " + "="*40)

        try:
            raw_image_name = sample.get("image_name", "")
            job_role = sample.get("job_role", "")
            
            pure_filename = os.path.basename(raw_image_name)
            
            base_dir = Path(__file__).resolve().parents[2]
            
            possible_paths = [
                os.path.join(base_dir, "app", "assets", "eval_data", pure_filename),
                os.path.join(base_dir, "app", "assests", "eval_data", pure_filename),
                os.path.join(base_dir, "assets", "eval_data", pure_filename),
                os.path.join(base_dir, "assests", "eval_data", pure_filename),
                os.path.join(os.getcwd(), "app", "assets", "eval_data", pure_filename),
                os.path.join(os.getcwd(), "app", "assests", "eval_data", pure_filename),
                os.path.join(os.getcwd(), "assets", "eval_data", pure_filename),
                os.path.join(os.getcwd(), "assests", "eval_data", pure_filename),
                os.path.join(base_dir, raw_image_name),
                os.path.join(os.getcwd(), raw_image_name)
            ]
            
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    break
                    
            if not img_path:
                raise FileNotFoundError(f"Could not locate {pure_filename} in any known asset directories.")

            gt_visual = sample.get("ground_truth_sub_audits", {}).get("visual_brand", {})
            true_range = gt_visual.get("true_score_range", {"min": 0.0, "max": 1.0})
            expected_flaws = gt_visual.get("expected_flaws", [])

            print(f" -> Successfully located image at: {img_path}")

            start_time = time()
            agent_output = self.invoke(image_path=img_path, job_role=job_role)
            end_time = time()
            invocation_time = end_time - start_time

            pred_score = agent_output.score
            pred_feedback = agent_output.feedback if agent_output.feedback else []

            print(f"[METRICS] True Range: [{true_range['min']} - {true_range['max']}] | Predicted: {pred_score}")
            print(f"[CONTENT] Expected Flaws: {expected_flaws}")
            print(f"[CONTENT] Agent Output: {pred_feedback}")

            midpoint = (true_range["min"] + true_range["max"]) / 2
            score_mae = abs(pred_score - midpoint)
            score_in_range = 1.0 if true_range["min"] <= pred_score <= true_range["max"] else 0.0

            matched_count = self._get_semantic_content_coverage(expected_flaws, pred_feedback)
            coverage_ratio = matched_count / len(expected_flaws) if expected_flaws else 1.0
            
            print(f"[EVAL RESULT] LLM Judge matched {matched_count}/{len(expected_flaws)} expected visual flaws.")
            print("-" * 100)

            return {
                "visual_brand_score_mae": round(score_mae, 4),
                "visual_brand_score_in_range_accuracy": score_in_range,
                "visual_brand_content_coverage_ratio": round(coverage_ratio, 4),
                "agent_invocation_time": round(invocation_time, 4)
            }

        except Exception as inner_error:
            print(f"\n[CRASH INSIDE VISUAL AGENT CASE #{self.case_counter}]: {inner_error}")
            traceback.print_exc()
            return {
                "visual_brand_score_mae": 0.0,
                "visual_brand_score_in_range_accuracy": 0.0,
                "visual_brand_content_coverage_ratio": 0.0,
                "agent_invocation_time": 0.0
            }