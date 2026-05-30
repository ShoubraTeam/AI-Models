import base64
import mimetypes
from agents.BaseAgent import BaseAgent
from schemas import VisualBrandEvaluationSchema
from helpers.config import DEFAULT_MODELS_CFG


class VisualBrandEvaluator(BaseAgent):

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["visual_brand_evaluator"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)

    def _encode_image_to_base64(self, image_path: str) -> tuple:

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"  
            
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
        return encoded_string, mime_type

    def invoke(self, image_path: str, job_role: str) -> VisualBrandEvaluationSchema:

        base64_image, mime_type = self._encode_image_to_base64(image_path)
        
        multimodal_content = [
            {
                "type": "text", 
                "text": (
                    f"Freelancer Job Role: {job_role}\n\n"
                    f"Please analyze this freelancer profile image directly and strictly according to your system prompt instructions, "
                    f"evaluating its appropriateness specifically for a {job_role}."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
            }
        ]
        
        return super().invoke(input=multimodal_content)
    

    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)