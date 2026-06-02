import re
from time import time
from agents.BaseAgent import BaseAgent
from helpers.config import DEFAULT_MODELS_CFG
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

class JobRequirementsExtractor(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["job_requirements_extractor"]


        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)
        self.case_counter = 0

    def get_agent(self):
        return super().get_agent()
    
    def invoke(self, input, return_structured_op_only = True):
        return super().invoke(input, return_structured_op_only)
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

    def _get_batch_semantic_matches(self, true_texts: list[str], pred_texts: list[str]) -> set[tuple[int, int]]:
        judge_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)
        
        true_list = "\n".join([f"LIST_A_{i}: {t}" for i, t in enumerate(true_texts)])
        pred_list = "\n".join([f"LIST_B_{j}: {p}" for j, p in enumerate(pred_texts)])
        
        prompt = f"""
        You are an expert NLP evaluation judge. Your task is to match identical or semantically equivalent requirements between List A (Ground Truth) and List B (Predicted Output).
        
        List A (Ground Truth):
        {true_list}
        
        List B (Predicted):
        {pred_list}
        
        Instructions:
        1. Compare items and find pairs that have the exact same functional meaning or technical scope.
        2. Output the matching pairs in this exact format: index_A-index_B (e.g., 0-0 or 1-2), one per line.
        3. CRITICAL: If there is only one item in List A and one item in List B and they match, output "0-0" immediately.
        4. Do NOT wrap the output in markdown code blocks. Do not write any explanations, intro, or outro text.
        """
        matches = set()
        try:
            response = judge_model.invoke([HumanMessage(content=prompt)])
            found_pairs = re.findall(r'(\d+)\s*-\s*(\d+)', response.content)
            
            for t_idx_str, p_idx_str in found_pairs:
                t_idx = int(t_idx_str)
                p_idx = int(p_idx_str)
                if t_idx < len(true_texts) and p_idx < len(pred_texts):
                    matches.add((t_idx, p_idx))
        except Exception as e:
            print(f" -> [BATCH JUDGE ERROR] {e}")
        return matches

    def get_metric_names(self) -> tuple[str, str, str, str, str]:
        return (
            "requirements_extraction_accuracy",
            "requirements_extraction_precision",
            "requirements_extraction_recall",
            "requirements_necessity_accuracy",
            "agent_invokation_time"
        )

    def evaluate_sample(self, sample: dict) -> dict:
        self.case_counter += 1 
        print(f"\n" + "="*50 + f" [DEBUG CASE #{self.case_counter}] " + "="*50)

        job_desc = sample["job_desc"]
        true_requirements = sample["job_data"].get("requirements", [])

        start_time = time()
        extracted_output = self.invoke(input = job_desc)
        end_time = time()
        
        pred_requirements = extracted_output.requirements           

        true_texts = [req.get("description", "") for req in true_requirements] 
        pred_texts = [getattr(req, "text", "") for req in pred_requirements]     
        
        print(f"True Requirements Count: {len(true_texts)} | Agent Extracted Count: {len(pred_texts)}")
        print(f"True Descriptions: {true_texts}")
        print(f"Pred Descriptions: {pred_texts}")
        print("-" * 115)

        matched_pairs = self._get_batch_semantic_matches(true_texts, pred_texts)

        matched_true = set()
        matched_pred = set()
        for t_idx, p_idx in matched_pairs:
            matched_true.add(t_idx)
            matched_pred.add(p_idx)
            print(f" -> [SEMANTIC MATCH] True #{t_idx} matched with Pred #{p_idx}")

        TP = len(matched_pred)                                    
        FP = len(pred_texts) - len(matched_pred)
        FN = len(true_texts) - len(matched_true)

        accuracy  = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        precision = TP / (TP + FP)      if (TP + FP)      else 0.0
        recall    = TP / (TP + FN)      if (TP + FN)      else 0.0

        correct_necessity = 0
        total_necessity = 0
        for t_idx, p_idx in matched_pairs:
            true_level = true_requirements[t_idx].get("necessity_level")
            pred_level = getattr(pred_requirements[p_idx], "necessity_level", "")
            total_necessity += 1
            if true_level == pred_level:
                correct_necessity += 1
            else:
                print(f" -> [NECESSITY MISMATCH] True #{t_idx} ({true_level}) VS Pred #{p_idx} ({pred_level})")

        necessity_acc = (correct_necessity / total_necessity) if total_necessity else 0.0

        print(f"[SAMPLE METRICS] Accuracy: {round(accuracy, 2)} | Precision: {round(precision, 2)} | Recall: {round(recall, 2)} | Necessity Accuracy: {round(necessity_acc, 2)}")

        return {
            "requirements_extraction_accuracy" : accuracy,
            "requirements_extraction_precision" : precision,
            "requirements_extraction_recall"    : recall,
            "requirements_necessity_accuracy"   : necessity_acc,
            "agent_invokation_time"            : end_time - start_time
        }