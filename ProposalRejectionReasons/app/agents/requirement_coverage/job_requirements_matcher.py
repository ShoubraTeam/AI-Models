from time import time

from helpers.config import DEFAULT_MODELS_CFG
from processing.requirement_coverage_processing import prepare_job_requirements_matcher_ip

from agents.BaseAgent import BaseAgent
from schemas import RequirementCoverageSchema

class JobRequirementsMatcher(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["job_requirements_matcher"]
        
        # kwargs["max_tokens"] = 4096

        super().__init__(model_name, system_prompt, structured_response, **kwargs)
        self.case_counter = 0

    # -------------------------------- Modeling ----------------------------- #
    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)


    def invoke(self, job_requirements: list, proposal_text: str) -> RequirementCoverageSchema:
        formatted_input = prepare_job_requirements_matcher_ip(
            job_requirements = job_requirements,
            proposal_text    = proposal_text
        )

        return super().invoke(input = formatted_input)


    async def ainvoke(self, job_requirements: list, proposal_text: str) -> RequirementCoverageSchema:
        formatted_input = prepare_job_requirements_matcher_ip(
            job_requirements = job_requirements,
            proposal_text    = proposal_text
        )

        return await super().ainvoke(input = formatted_input)
    
    # ---------------------------- Evaluation ----------------------------- #
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
        print(f"\n" + "="*50 + f" [DEBUG MATCHER CASE #{self.case_counter}] " + "="*50)

        import traceback
        try:
            job_data = sample.get("job_data", {})
            job_requirements = job_data.get("requirements", []) 
            
            sample_accuracy = []
            sample_precision = []
            sample_recall = []
            sample_f1 = []
            total_invocation_time = 0.0

            proposals = sample.get("proposals", [])
            for p_sample in proposals:
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

                all_ids = true_covered.union(true_missing).union(pred_covered).union(pred_missing)
                
                if not all_ids:
                    print(" -> [SKIP] No IDs found in ground truth or predictions for this sample.")
                    print("-" * 114)
                    continue

                tp, fp, fn, tn = 0, 0, 0, 0

                for cid in all_ids:
                    is_true_covered = cid in true_covered
                    is_pred_covered = cid in pred_covered

                    if is_true_covered:
                        if is_pred_covered:
                            tp += 1
                        else:
                            fn += 1
                    else:  
                        if is_pred_covered:
                            fp += 1
                        else:
                            tn += 1

                print(f"[COUNTS]      TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

                total = tp + fp + fn + tn
                accuracy = (tp + tn) / total if total > 0 else 0.0
                
                if len(true_covered) == 0 and len(pred_covered) == 0:
                    precision = 1.0
                    recall = 1.0
                else:
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
            
        except Exception as e:
            print(f"\n[MATCHER CRASH DETECTED INSIDE CASE #{self.case_counter} !!!]")
            traceback.print_exc()
            return {k: 0.0 for k in self.get_metric_names()}