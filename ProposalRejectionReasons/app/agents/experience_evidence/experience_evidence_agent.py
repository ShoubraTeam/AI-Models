import os
import re
from time import time

from helpers.config import DEFAULT_MODELS_CFG
from processing.experience_evidence import prepare_experience_evidence_evaluator_ip

from groq_core import GroqModelsAPI
from agents.BaseAgent import BaseAgent
from schemas.experience_evidence import ExperienceEvidenceSchema

class ExperienceEvidenceAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["experience_evidence_agent"]

        super().__init__(model_name, system_prompt, structured_response, **kwargs)
        self.case_counter = 0

    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)
    
    # ------------------------------ Calling -------------------------- #
    
    def invoke(self, job_desc: str, proposal_text: str) -> ExperienceEvidenceSchema:
        formatted_input = prepare_experience_evidence_evaluator_ip(job_desc, proposal_text)

        return super().invoke(input = formatted_input)


    async def ainvoke(self, job_desc: str, proposal_text: str) -> ExperienceEvidenceSchema:
        formatted_input = prepare_experience_evidence_evaluator_ip(job_desc, proposal_text)
        return await super().ainvoke(input=formatted_input)
    
    # ------------------------------------------- Evaluation ------------------------------------------- #
    def _get_semantic_project_match(self, true_texts: list[str], pred_texts: list[str]) -> set[tuple[int, int]]:
        if not true_texts or not pred_texts:
            return set()

        judge_model = GroqModelsAPI(api_key=os.getenv("GROQ_API_KEY"))
        
        true_list = "\n".join([f"TRUE_{i}: {t}" for i, t in enumerate(true_texts)])
        pred_list = "\n".join([f"PRED_{j}: {p}" for j, p in enumerate(pred_texts)])
        
        prompt = f"""
        You are an expert technical judge. Your task is to match identical or semantically equivalent past projects between List True (Ground Truth) and List Pred (Agent Output).
        
        List True (Ground Truth):
        {true_list}
        
        List Pred (Agent Predicted):
        {pred_list}
        
        Instructions:
        1. Compare items and find pairs that refer to the same past project or experience, even if described with different words.
        2. Output the matching pairs in this exact format: index_True-index_Pred (e.g., 0-0 or 1-2), one per line.
        3. If there is only one item in List True and one item in List Pred and they refer to the same project context, output "0-0" immediately.
        4. Do NOT write any markdown blocks, introductions, or explanations. Output ONLY the raw pairs.
        """
        matches = set()
        try:
            response = judge_model.generate(
                model_name="llama-3.1-8b-instant",
                user_input=prompt,
                temperature=0.0,
                timeout=30,
            )
            found_pairs = re.findall(r'(\d+)\s*-\s*(\d+)', response)
            
            for t_idx_str, p_idx_str in found_pairs:
                t_idx = int(t_idx_str)
                p_idx = int(p_idx_str)
                if t_idx < len(true_texts) and p_idx < len(pred_texts):
                    matches.add((t_idx, p_idx))
        except Exception as e:
            print(f" -> [JUDGE ERROR] {e}")
        return matches

    def get_metric_names(self) -> tuple[str, str, str, str, str, str, str, str]:
        return (
            "classification_accuracy",
            "classification_precision",
            "classification_recall",
            "classification_f1_score",
            "project_extraction_precision",
            "project_extraction_recall",
            "project_score_mae",
            "agent_invokation_time"
        )

    def evaluate_sample(self, sample: dict) -> dict:
        self.case_counter += 1 
        print(f"\n" + "="*50 + f" [DEBUG EXPERIENCE CASE #{self.case_counter}] " + "="*50)

        job_desc = sample.get("job_desc", "")
        proposals = sample.get("proposals", [])

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        total_tp_projects = 0
        total_pred_projects = 0
        total_true_projects = 0
        all_score_errors = []
        total_invocation_time = 0.0

        for prop_sample in proposals:
            proposal_text = prop_sample.get("proposal", "")
            true_has_evidence = prop_sample.get("has_evidence", False)
            true_projects = prop_sample.get("true_projects", [])

            start_time = time()
            agent_output = self.invoke(job_desc, proposal_text)
            end_time = time()
            total_invocation_time += (end_time - start_time)

            pred_has_evidence = agent_output.has_experience_evidence
            pred_projects = agent_output.extracted_projects if agent_output.extracted_projects else []

            true_texts = [p.get("project_overview", "") if isinstance(p, dict) else str(p) for p in true_projects]
            pred_texts = [p.project_overview for p in pred_projects]

            print(f"[CLASSIFICATION] True Has Evidence: {true_has_evidence} | Agent Predicted: {pred_has_evidence}")
            print(f"[PROJECTS]       True Projects: {true_texts}")
            print(f"[PROJECTS]       Agent Extracted: {pred_texts}")

            if true_has_evidence and pred_has_evidence:
                tp += 1
            elif not true_has_evidence and pred_has_evidence:
                fp += 1
            elif true_has_evidence and not pred_has_evidence:
                fn += 1
            elif not true_has_evidence and not pred_has_evidence:
                tn += 1

            total_pred_projects += len(pred_projects)
            total_true_projects += len(true_projects)

            matched_pairs = self._get_semantic_project_match(true_texts, pred_texts)

            for t_idx, p_idx in matched_pairs:
                total_tp_projects += 1
                
                true_proj_data = true_projects[t_idx]
                if isinstance(true_proj_data, dict) and "relevance_score" in true_proj_data:
                    true_score = true_proj_data.get("relevance_score", 0.0)
                    all_score_errors.append(abs(pred_projects[p_idx].relevance_score - true_score))
                
                print(f"    [MATCH FOUND] True Project #{t_idx} matched with Pred Project #{p_idx} via LLM Judge")

            matched_pred_set = {p_idx for _, p_idx in matched_pairs}
            for p_idx in range(len(pred_texts)):
                if p_idx not in matched_pred_set:
                    print(f"    [NO MATCH] Pred Project Index {p_idx} had no semantic equivalence in ground truth")
            
            print("-" * 114)

        total_samples = tp + fp + fn + tn
        class_acc = (tp + tn) / total_samples if total_samples > 0 else 0.0
        class_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        class_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_f1 = (2 * class_prec * class_rec) / (class_prec + class_rec) if (class_prec + class_rec) > 0 else 0.0

        proj_precision = total_tp_projects / total_pred_projects if total_pred_projects > 0 else 1.0 if not total_true_projects and not total_pred_projects else 0.0
        proj_recall = total_tp_projects / total_true_projects if total_true_projects > 0 else 1.0 if not total_true_projects and not total_pred_projects else 0.0
        proj_mae = sum(all_score_errors) / len(all_score_errors) if all_score_errors else 0.0

        return {
            "classification_accuracy": class_acc,
            "classification_precision": class_prec,
            "classification_recall": class_rec,
            "classification_f1_score": class_f1,
            "project_extraction_precision": proj_precision,
            "project_extraction_recall": proj_recall,
            "project_score_mae": proj_mae,
            "agent_invokation_time": total_invocation_time
        }
