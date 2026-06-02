import json
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
    
    def evaluate(self, eval_data: list[dict]) -> dict:
        metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        }

        for idx, sample in enumerate(eval_data, start=1):
            job_data = sample.get("job_data", {})
            job_requirements = job_data.get("requirements", []) 
            
            for p_sample in sample.get("proposals", []):
                proposal_text = p_sample.get("proposal", "")
                
                true_covered = set(str(cid).strip().lower() for cid in p_sample.get("true_covered_ids", []))
                true_missing = set(str(mid).strip().lower() for mid in p_sample.get("true_missing_ids", []))

                agent_output = self.invoke(job_requirements, proposal_text)
                
                pred_covered = set(str(cid).strip().lower() for cid in getattr(agent_output, "requirements_covered_ids", []))
                pred_missing = set(str(mid).strip().lower() for mid in getattr(agent_output, "missing_requirements_ids", []))

                print(f"\n" + "="*50 + f" [DEBUG MATCHER CASE #{idx}] " + "="*50)
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
                print("-" * 114)

                total = tp + fp + fn + tn
                accuracy = (tp + tn) / total if total > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                metrics["accuracy"].append(accuracy)
                metrics["precision"].append(precision)
                metrics["recall"].append(recall)
                metrics["f1_score"].append(f1)

        if not metrics["accuracy"]:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        return {
            "accuracy": round(sum(metrics["accuracy"]) / len(metrics["accuracy"]), 4),
            "precision": round(sum(metrics["precision"]) / len(metrics["precision"]), 4),
            "recall": round(sum(metrics["recall"]) / len(metrics["recall"]), 4),
            "f1_score": round(sum(metrics["f1_score"]) / len(metrics["f1_score"]), 4)
        }