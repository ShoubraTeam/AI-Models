from agents.BaseAgent import BaseAgent
from schemas.experience_evidence import ExperienceEvidenceSchema
from helpers.config import DEFAULT_MODELS_CFG

class ExperienceEvidenceAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["experience_evidence_agent"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)

    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

    def invoke(self, job_desc: str, proposal_text: str) -> ExperienceEvidenceSchema:
        formatted_input = f"Job Description:\n{job_desc}\n\nFreelancer Proposal Text:\n{proposal_text}"
        return super().invoke(input=formatted_input)

    def _calc_text_similarity(self, str1: str, str2: str) -> float:
        import re
        
        tokens1 = set(re.findall(r'[a-z0-9]+', str1.lower()))
        tokens2 = set(re.findall(r'[a-z0-9]+', str2.lower()))
        
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'for', 'with', 'on', 'in', 'at', 'by', 'from', 'to',
            'built', 'developed', 'designed', 'store', 'website', 'platform', 'freelancer', 'project', 
            'similar', 'custom', 'responsive', 'ecommerce', 'e', 'commerce', 'co', 'uk', 'com', 'https', 
            'http', 'implementation', 'development', 'redesign', 'customization', 'storefront', 'app',
            'pages', 'site', 'another', 'features', 'templated', 'focusing', 'subagent'
        }
        
        unique_pred = tokens1 - stopwords
        unique_true = tokens2 - stopwords
        
        if not unique_true:
            return 0.0
            
        intersection = unique_pred.intersection(unique_true)
        
        return float(len(intersection) / len(unique_true))

    def evaluate(self, eval_data: list[dict]) -> dict:
        metrics = {
            "classification_accuracy": 0.0,
            "classification_precision": 0.0,
            "classification_recall": 0.0,
            "classification_f1_score": 0.0,
            "project_extraction_precision": 0.0,
            "project_extraction_recall": 0.0,
            "project_score_mae": 0.0
        }

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        total_tp_projects = 0
        total_pred_projects = 0
        total_true_projects = 0
        all_score_errors = []

        for job_sample in eval_data:
            job_desc = job_sample["job_desc"]
            proposals = job_sample.get("proposals", [])

            for prop_sample in proposals:
                proposal_text = prop_sample["proposal"]
                true_has_evidence = prop_sample["has_evidence"]
                true_projects = prop_sample.get("true_projects", [])

                agent_output = self.invoke(job_desc, proposal_text)
                pred_has_evidence = agent_output.has_experience_evidence
                pred_projects = agent_output.extracted_projects if agent_output.extracted_projects else []

                print(f"\n" + "="*50 + " [DEBUG CASE] " + "="*50)
                print(f"[CLASSIFICATION] True Has Evidence: {true_has_evidence} | Agent Predicted: {pred_has_evidence}")
                print(f"[PROJECTS]       True Projects: {[p.get('project_overview') for p in true_projects]}")
                print(f"[PROJECTS]       Agent Extracted: {[p.project_overview for p in pred_projects]}")
                print("-" * 114)

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

                all_possible_pairs = []
                for p_idx, pred_proj in enumerate(pred_projects):
                    for t_idx, true_proj in enumerate(true_projects):
                        sim = self._calc_text_similarity(pred_proj.project_overview, true_proj.get("project_overview", ""))
                        all_possible_pairs.append((sim, p_idx, t_idx))

                all_possible_pairs.sort(key=lambda x: x[0], reverse=True)

                matched_pred_indices = set()
                matched_true_indices = set()

                for sim, p_idx, t_idx in all_possible_pairs:
                    if p_idx in matched_pred_indices or t_idx in matched_true_indices:
                        continue
                    
                    if sim >= 0.35:
                        matched_pred_indices.add(p_idx)
                        matched_true_indices.add(t_idx)
                        total_tp_projects += 1
                        true_score = true_projects[t_idx].get("relevance_score", 0.0)
                        all_score_errors.append(abs(pred_projects[p_idx].relevance_score - true_score))
                        print(f"    [MATCH FOUND] Pred Project Index {p_idx} matched with True Project Index {t_idx} (Sim={round(sim, 2)})")
                    else:
                        print(f"    [NO MATCH] Best remaining similarity was {round(sim, 2)} (Below 0.35 Threshold)")

        total_samples = tp + fp + fn + tn
        if total_samples > 0:
            metrics["classification_accuracy"] = round((tp + tn) / total_samples, 4)
        if (tp + fp) > 0:
            metrics["classification_precision"] = round(tp / (tp + fp), 4)
        if (tp + fn) > 0:
            metrics["classification_recall"] = round(tp / (tp + fn), 4)
        
        p = metrics["classification_precision"]
        r = metrics["classification_recall"]
        if (p + r) > 0:
            metrics["classification_f1_score"] = round((2 * p * r) / (p + r), 4)

        if total_pred_projects > 0:
            metrics["project_extraction_precision"] = round(total_tp_projects / total_pred_projects, 4)
        if total_true_projects > 0:
            metrics["project_extraction_recall"] = round(total_tp_projects / total_true_projects, 4)
        if all_score_errors:
            metrics["project_score_mae"] = round(sum(all_score_errors) / len(all_score_errors), 4)

        return metrics