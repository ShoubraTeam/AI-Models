from agents.BaseAgent import BaseAgent
from schemas import SuperAgentSchema
import json

class SuperAgent(BaseAgent):
    """
    The Master Orchestrator Agent. Consumes outputs from all other sub-agents,
    performs cross-domain reasoning, and compiles the final Executive Audit Report.
    """
    def __init__(self, model_name: str, system_prompt: str, tools: list = [], structured_response = None, **kwargs):
        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)

    def invoke(self, visual_res, bio_res, skills_res, numerical_res) -> SuperAgentSchema:
        
        def to_dict(obj):
            if hasattr(obj, "dict"): return obj.dict()
            if hasattr(obj, "model_dump"): return obj.model_dump()
            return obj

        formatted_input = (
            "=== SUB-AUDIT 1: VISUAL BRAND ===\n"
            f"{json.dumps(to_dict(visual_res), indent=2)}\n\n"
            "=== SUB-AUDIT 2: BIO COPYWRITING ===\n"
            f"{json.dumps(to_dict(bio_res), indent=2)}\n\n"
            "=== SUB-AUDIT 3: SKILLS ALIGNMENT ===\n"
            f"{json.dumps(to_dict(skills_res), indent=2)}\n\n"
            "=== SUB-AUDIT 4: NUMERICAL METRICS ENGINE ===\n"
            f"{json.dumps(to_dict(numerical_res), indent=2)}\n"
        )
        
        return super().invoke(input=formatted_input)