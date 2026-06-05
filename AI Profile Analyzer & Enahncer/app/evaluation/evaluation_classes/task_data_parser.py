class EvaluationDataParser:
    """
    A utility class strictly refactored for the Freelancer Profile Auditor Suite.
    Extracts the exact target fields required by each agent from the unified dataset.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_visual_data(sample: dict) -> dict:
        """Extracts inputs required for the VisualBrandEvaluator."""
        return {
            "image_name": sample.get("image_name", ""),
            "job_role": sample.get("job_role", "")
        }

    @staticmethod
    def get_bio_data(sample: dict) -> dict:
        """Extracts inputs required for the BioAnalyzer."""
        return {
            "bio_text": sample.get("bio_text", ""),
            "job_role": sample.get("job_role", "")
        }

    @staticmethod
    def get_skills_data(sample: dict) -> dict:
        """Extracts inputs required for the SkillsAnalyzer."""
        return {
            "declared_skills": sample.get("declared_skills", []),
            "job_role": sample.get("job_role", "")
        }

    @staticmethod
    def get_numerical_data(sample: dict) -> dict:
        """Extracts inputs required for the Rule-Based NumericalAnalyzer Engine."""
        return {
            "job_role": sample.get("job_role", ""),
            "hourly_rate": sample.get("hourly_rate", 0.0),
            "rating": sample.get("rating", 0.0),
            "total_completed_jobs": sample.get("total_completed_jobs", 0)
        }

    @staticmethod
    def get_super_agent_data(sample: dict) -> dict:
        """
        Extracts high-level context for the SuperAgent/Master Orchestrator
        to log metadata or map the final consolidated profile entity.
        """
        return {
            "freelancer_name": sample.get("freelancer_name", ""),
            "job_role": sample.get("job_role", "")
        }