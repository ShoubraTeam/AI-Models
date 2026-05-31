from schemas import NumericalAnalyzerSchema

class NumericalAnalyzer:
    """
    Granular Rule-Based version of the Numerical Analyzer.
    Evaluates pricing bounds specifically for each of the 9 individual roles.
    """
    def __init__(self, *args, **kwargs):
        pass

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