from time import time
from schemas import NumericalAnalyzerSchema

class NumericalAnalyzer:
    """
    Granular Rule-Based version of the Numerical Analyzer.
    Evaluates pricing bounds specifically for each of the 9 individual roles.
    Includes built-in pure Python deterministic evaluation execution.
    """
    def __init__(self, *args, **kwargs):
        self.case_counter = 0

    def invoke(
        self,
        job_role: str,
        hourly_rate: float,
        rating: float,
        total_completed_jobs: int
    ) -> NumericalAnalyzerSchema:
        
        if total_completed_jobs < 5:
            seniority = "Beginner"
        elif 5 <= total_completed_jobs <= 25:
            seniority = "Mid-Tier"
        else:
            seniority = "Expert"
            
        pricing_matrix = {
            "AI Engineer": {
                "Beginner": {"min": 35, "max": 60},
                "Mid-Tier": {"min": 65, "max": 110},
                "Expert": {"min": 120, "max": 250}
            },
            "DevOps Engineer": {
                "Beginner": {"min": 30, "max": 50},
                "Mid-Tier": {"min": 55, "max": 90},
                "Expert": {"min": 100, "max": 180}
            },
            "Backend Developer": {
                "Beginner": {"min": 25, "max": 40},
                "Mid-Tier": {"min": 45, "max": 75},
                "Expert": {"min": 80, "max": 150}
            },
            "Mobile Developer": {
                "Beginner": {"min": 25, "max": 40},
                "Mid-Tier": {"min": 45, "max": 70},
                "Expert": {"min": 75, "max": 140}
            },
            "Frontend Developer": {
                "Beginner": {"min": 20, "max": 35},
                "Mid-Tier": {"min": 40, "max": 60},
                "Expert": {"min": 65, "max": 120}
            },
            "Data Analyst": {
                "Beginner": {"min": 20, "max": 35},
                "Mid-Tier": {"min": 40, "max": 65},
                "Expert": {"min": 70, "max": 130}
            },
            "Video Producer": {
                "Beginner": {"min": 20, "max": 35},
                "Mid-Tier": {"min": 40, "max": 60},
                "Expert": {"min": 65, "max": 110}
            },
            "Graphic Designer": {
                "Beginner": {"min": 15, "max": 25},
                "Mid-Tier": {"min": 30, "max": 50},
                "Expert": {"min": 55, "max": 100}
            },
            "Content Writer": {
                "Beginner": {"min": 15, "max": 25},
                "Mid-Tier": {"min": 30, "max": 45},
                "Expert": {"min": 50, "max": 90}
            }
        }
        
        default_bounds = {"min": 20, "max": 50}
        role_bounds = pricing_matrix.get(job_role, {}).get(seniority, default_bounds)
        
        if hourly_rate > role_bounds["max"]:
            pricing_status = "Overpriced"
        elif hourly_rate < role_bounds["min"]:
            pricing_status = "Underpriced"
        else:
            pricing_status = "Fair Market Value"
            
        improvements = []
        base_score = 1.0
        
        if pricing_status == "Overpriced":
            base_score -= 0.4
            improvements.append(f"Decrease hourly rate to ${role_bounds['max']} to match global {job_role} {seniority} standards.")
        elif pricing_status == "Underpriced":
            base_score -= 0.2
            improvements.append(f"Increase hourly rate to at least ${role_bounds['min']} to match your high value as a {seniority} {job_role}.")
            
        if rating < 4.7:
            base_score -= 0.3
            improvements.append("Improve client satisfaction to push your star rating above 4.7.")
        
        if total_completed_jobs < 5:
            improvements.append("Complete more projects to safely transition into higher pricing tiers.")
            
        final_score = max(0.0, min(1.0, round(base_score, 2)))
        
        if not improvements:
            improvements.append(f"Your pricing and metrics are perfectly optimized for a {seniority} {job_role}.")

        return NumericalAnalyzerSchema(
            score=final_score,
            pricing_status=pricing_status,
            improvements=improvements,
            confidence_score=1.0
        )

    def get_metric_names(self) -> tuple[str, str, str]:
        return (
            "numerical_score_mae",
            "pricing_status_accuracy",
            "agent_invocation_time"
        )

    def evaluate_sample(self, sample: dict) -> dict:
        """
        Evaluates the pure Python rule-based engine output against Ground Truth.
        Compares numeric metrics instantly with 100% determinism.
        """
        self.case_counter += 1 
        print(f"\n" + "="*40 + f" [EVALUATING NUMERICAL CASE #{self.case_counter}] " + "="*40)

        job_role = sample.get("job_role", "")
        hourly_rate = sample.get("hourly_rate", 0.0)
        rating = sample.get("rating", 0.0)
        total_completed_jobs = sample.get("total_completed_jobs", 0)

        gt_numerical = sample.get("ground_truth_sub_audits", {}).get("numerical_metrics", {})
        true_score = gt_numerical.get("true_score", 1.0)
        true_status = gt_numerical.get("true_pricing_status", "Fair Market Value")

        start_time = time()
        agent_output = self.invoke(
            job_role=job_role,
            hourly_rate=hourly_rate,
            rating=rating,
            total_completed_jobs=total_completed_jobs
        )
        end_time = time()
        invocation_time = end_time - start_time

        pred_score = agent_output.score
        pred_status = agent_output.pricing_status

        print(f"[METRICS] True Score: {true_score} | Agent Output Score: {pred_score}")
        print(f"[STATUS]  True Status: '{true_status}' | Agent Output Status: '{pred_status}'")

        score_mae = abs(pred_score - true_score)
        status_accuracy = 1.0 if pred_status.strip().lower() == true_status.strip().lower() else 0.0

        print(f"[EVAL RESULT] Status Accuracy: {status_accuracy} | Invocation Time: {round(invocation_time, 6)}s")
        print("-" * 100)

        return {
            "numerical_score_mae": round(score_mae, 4),
            "pricing_status_accuracy": status_accuracy,
            "agent_invocation_time": round(invocation_time, 6)
        }