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
        1. Compare items and find pairs that have the same functional meaning or technical scope.
        2. CRITICAL: Multiple items from List A can map to a single item in List B. If one comprehensive requirement in List B covers several specific requirements in List A, map all of those List A indices to that same List B index.
        3. Output the matching pairs in this exact format: index_A-index_B (e.g., 0-0, 1-0, 2-0), one per line.
        4. Do NOT wrap the output in markdown code blocks. Do not write any explanations.
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

        import traceback
        try:
            job_desc = sample.get("job_desc", "")
            job_data = sample.get("job_data", {})
            true_requirements = job_data.get("requirements", [])

            start_time = time()
            extracted_output = self.invoke(input = job_desc)
            end_time = time()
            
            pred_requirements = extracted_output.requirements[:10]           

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

            precision = len(matched_pred) / len(pred_texts) if pred_texts else 0.0
            recall = len(matched_true) / len(true_texts) if true_texts else 0.0
            
            union_size = len(true_texts) + len(pred_texts) - len(matched_pred)
            accuracy = len(matched_true) / union_size if union_size > 0 else 0.0

            correct_necessity = 0
            total_necessity = 0
            for t_idx, p_idx in matched_pairs:
                true_level = true_requirements[t_idx].get("necessity_level", "mandatory") 
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
            
        except Exception as e:
            print(f"\n[EXTRACTOR CRASH DETECTED INSIDE CASE #{self.case_counter} !!!]")
            traceback.print_exc()
            return {k: 0.0 for k in self.get_metric_names()}