# ---------------------------------------------------------------------
# A utility class used to init the agents required for a specific task
# ---------------------------------------------------------------------

# agents
from agents import NumericalAnalyzer, BioAnalyzer, SkillsAnalyzer, VisualBrandEvaluator, SuperAgent



# response schemas
from schemas import  VisualBrandEvaluationSchema, BioAnalyzerSchema, SkillsAnalyzerSchema, NumericalAnalyzerSchema, SuperAgentSchema

# system prompts
from prompts import VISUAL_BRAND_PROMPT, BIO_ANALYZER_PROMPT, SKILLS_ANALYZER_PROMPT, SUPER_AGENT_PROMPT

class AgentsInitializer:
    """
    A class used to init the agents required to evaluate them on a specific task
    """

    def __init__(self):
        pass

    @staticmethod
    def get_numerical_analyzer_agent(model_name, **kwargs):
        return NumericalAnalyzer(
            model_name          = model_name,
            system_prompt       = "none",
            structured_response = NumericalAnalyzerSchema
        )

    @staticmethod
    def get_bio_analyzer_agent(model_name, **kwargs):
        return BioAnalyzer(
            model_name          = model_name,
            system_prompt       = BIO_ANALYZER_PROMPT,
            structured_response = BioAnalyzerSchema,
        )

    @staticmethod
    def get_skills_analyzer_agent(model_name, **kwargs):
        return SkillsAnalyzer(
            model_name          = model_name, 
            system_prompt       = SKILLS_ANALYZER_PROMPT,
            structured_response = SkillsAnalyzerSchema,
        )

    @staticmethod
    def get_visual_brand_evaluator_agent(model_name, **kwargs):
        return VisualBrandEvaluator(
            model_name          = model_name, 
            system_prompt       = VISUAL_BRAND_PROMPT,
            structured_response = VisualBrandEvaluationSchema,
        )

    # super-agent
    @staticmethod
    def get_super_agent(model_name, **kwargs):
        return SuperAgent(
            model_name          = model_name, 
            system_prompt       = SUPER_AGENT_PROMPT,
            structured_response = SuperAgentSchema
        )


    