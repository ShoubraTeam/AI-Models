import json
from time import time
from agents.BaseAgent import BaseAgent
from schemas import RequirementCoverageSchema
from helpers.config import DEFAULT_MODELS_CFG

class JobRequirementsMatcher(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["job_requirements_matcher"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)
        self.case_counter = 0

    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

    def invoke(self, job_requirements: list[dict], proposal_text: str) -> RequirementCoverageSchema:
        mapped_requirements = []
        for req in job_requirements:
            mapped_requirements.append({
                "id": req.get("id"),
                "text": req.get("description", req.get("text", "")), 
                "necessity_level": req.get("necessity_level")
            })
        
        requirements_json = json.dumps(mapped_requirements, indent=2)
        formatted_input = f"Job Requirements List:\n{requirements_json}\n\nFreelancer Proposal Text:\n{proposal_text}"
        return super().invoke(input=formatted_input)
    
    def get_metric_names(self) -> tuple[str, str, str, str, str]:
        return (
            "matcher_accuracy",
            "matcher_precision",
            "matcher_recall",
            "matcher_f1_score",
            "agent_invokation_time"
        )
    
    def evaluate_sample(self, sample: dict) -> dict:
        self.case_counter += 1 
        print(f"\n" + "="*50 + f" [DEBUG CASE #{self.case_counter}] " + "="*50)
        job_data = sample.get("job_data", {})
        job_requirements = job_data.get("requirements", []) 
        
        sample_accuracy = []
        sample_precision = []
        sample_recall = []
        sample_f1 = []
        total_invocation_time = 0.0

        proposals = sample.get("proposals", [])
        for idx, p_sample in enumerate(proposals, start=1):
            proposal_text = p_sample.get("proposal", "")
            
            true_covered = set(str(cid).strip().lower() for cid in p_sample.get("true_covered_ids", []))
            true_missing = set(str(mid).strip().lower() for mid in p_sample.get("true_missing_ids", []))

            start_time = time()
            agent_output = self.invoke(job_requirements, proposal_text)
            end_time = time()
            total_invocation_time += (end_time - start_time)
            
            pred_covered = set(str(cid).strip().lower() for cid in getattr(agent_output, "requirements_covered_ids", []))
            pred_missing = set(str(mid).strip().lower() for mid in getattr(agent_output, "missing_requirements_ids", []))

            print(f"[COVERED IDs] True: {true_covered} | Agent Predicted: {pred_covered}")
            print(f"[MISSING IDs] True: {true_missing} | Agent Predicted: {pred_missing}")

            all_ids = true_covered.union(true_missing)
            if not all_ids:
                print(" -> [SKIP] No IDs found in ground truth for this sample.")
                print("-" * 114)
                continue

            tp = len(pred_covered.intersection(true_covered))
            fp = len(pred_covered.intersection(true_missing))
            fn = len(pred_missing.intersection(true_covered))
            tn = len(pred_missing.intersection(true_missing))

            print(f"[COUNTS]      TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

            total = tp + fp + fn + tn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            sample_accuracy.append(accuracy)
            sample_precision.append(precision)
            sample_recall.append(recall)
            sample_f1.append(f1)

            print(f"[SAMPLE METRICS] Accuracy: {round(accuracy, 2)} | Precision: {round(precision, 2)} | Recall: {round(recall, 2)} | F1: {round(f1, 2)}")
            print("-" * 114)

        num_proposals = len(sample_accuracy) if sample_accuracy else 1
        
        return {
            "matcher_accuracy": sum(sample_accuracy) / num_proposals,
            "matcher_precision": sum(sample_precision) / num_proposals,
            "matcher_recall": sum(sample_recall) / num_proposals,
            "matcher_f1_score": sum(sample_f1) / num_proposals,
            "agent_invokation_time": total_invocation_time
        }