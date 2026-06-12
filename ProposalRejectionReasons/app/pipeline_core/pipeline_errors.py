# --------------------------------------------------
# Define Errors that the system may face
# --------------------------------------------------


class ProposalRejecionReasonsError(Exception):
    default_message = "Proposal rejection reasons error"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.default_message)


class JobToolsExtractorError(ProposalRejecionReasonsError):
    default_message = "Job Tools Extractor Error"

class ProposalToolsAnalyzerError(ProposalRejecionReasonsError):
    default_message = "Proposal Tools Analyzer Error"

class JobKeyPointsExtractorError(ProposalRejecionReasonsError):
    default_message = "Job Key Points Extractor Error"

class JobUnderstandingEvaluatorError(ProposalRejecionReasonsError):
    default_message = "Job Understanding Evaluator Error"

class JobRequirementExtractorError(ProposalRejecionReasonsError):
    default_message = "Job Requirement Extractor Error"

class RequirmentCoverageEvaluatorError(ProposalRejecionReasonsError):
    default_message = "Requirment Coverage Evaluator Error"

class ExperienceEvidenceEvaluatorError(ProposalRejecionReasonsError):
    default_message = "Experience Evidence Evaluator Error"

class LanguageClarityEvaluatorError(ProposalRejecionReasonsError):
    default_message = "Language Clarity Evaluator Error"
