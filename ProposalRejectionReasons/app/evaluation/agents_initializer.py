# ---------------------------------------------------------------------
# A utility class used to init the agents required for a specific task
# ---------------------------------------------------------------------
from agents import ExperienceEvidenceAgent
from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator
from agents import JobRequirementsExtractor, JobRequirementsMatcher

class AgentsInitializer:
    """
    A class used to init the agents required to evaluate them on a specific task
    """

    def __init__(self):
        pass

    @staticmethod
    def get_tools_alignment_agents(models, system_prompts, structured_responses, **kwargs):
        job_tools_extractor = JobToolsExtractor(
            model_name = models[0],
            system_prompt = system_prompts[0],
            structured_response = structured_responses[0],
            **kwargs
        )

        proposal_tools_analyzer = ProposalToolsAnalyzer(
            model_name = models[1],
            system_prompt = system_prompts[1],
            structured_response = structured_responses[1],
            **kwargs
        )

        agents = [
            ("job_tools_extractor"     , job_tools_extractor),
            ("proposal_tools_analyzer" , proposal_tools_analyzer)
        ]

        return agents
    


    @staticmethod
    def get_job_understanding_agents(models, system_prompts, structured_responses, **kwargs):
        job_key_points_extractor = JobKeyPointsExtractor(
            model_name = models[0],
            system_prompt = system_prompts[0],
            structured_response = structured_responses[0],
            **kwargs
        )

        job_understanding_evaluator = JobUnderstandingEvaluator(
            model_name = models[1],
            system_prompt = system_prompts[1],
            structured_response = structured_responses[1],
            **kwargs
        )

        agents = [
            ("job_key_points_extractor"    , job_key_points_extractor),
            ("job_understanding_evaluator" , job_understanding_evaluator)
        ]

        return agents
    


    @staticmethod
    def get_requirement_coverage_agents(models, system_prompts, structured_responses, **kwargs):
        job_requirements_extractor = JobRequirementsExtractor(
            model_name = models[0],
            system_prompt = system_prompts[0],
            structured_response = structured_responses[0],
            **kwargs
        )

        job_requirements_matcher = JobRequirementsMatcher(
            model_name = models[1],
            system_prompt = system_prompts[1],
            structured_response = structured_responses[1],
            **kwargs
        )

        agents = [
            ("job_requirements_extractor", job_requirements_extractor),
            ("job_requirements_matcher"  , job_requirements_matcher)
        ]

        return agents
    

    @staticmethod
    def get_evidence_of_experience_agents(models, system_prompts, structured_responses, **kwargs):
        experience_evidence_agent = ExperienceEvidenceAgent(
            model_name = models[0],
            system_prompt = system_prompts[0],
            structured_response = structured_responses[0],
            **kwargs
        )

        agents = [
            ("experience_evidence_agent", experience_evidence_agent)
        ]
        return agents

    @staticmethod
    def get_language_clarity_agents(models, system_prompts, structured_responses, **kwargs):
        pass

    @staticmethod
    def get_super_agent(models, system_prompts, structured_responses, **kwargs):
        pass