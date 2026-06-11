# --------------------------------------------------
# Profile Scorer Pipeline Errors
# --------------------------------------------------

class ProfileScorerError(Exception):
    default_message = "Profile scorer pipeline error"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.default_message)


class NumericalAnalyzerError(ProfileScorerError):
    default_message = "Numerical Analyzer Error"


class BioAnalyzerError(ProfileScorerError):
    default_message = "Bio Analyzer Error"


class SkillsAnalyzerError(ProfileScorerError):
    default_message = "Skills Analyzer Error"


class VisualBrandEvaluatorError(ProfileScorerError):
    default_message = "Visual Brand Evaluator Error"


class ProfileSuperAgentError(ProfileScorerError):
    default_message = "Profile Super Agent Orchestrator Error"