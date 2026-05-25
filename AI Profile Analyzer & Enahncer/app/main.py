# ---------------------------------------------------------------------
# The Main Workflow - Freelancer Profile Auditor (Visual Brand Only)
# ---------------------------------------------------------------------

# agents
from agents import VisualBrandEvaluator

# schemas
from schemas import VisualBrandEvaluationSchema

# prompts
from prompts import VISUAL_BRAND_PROMPT

# others
import os
from pathlib import Path
import helpers.config as CFG
import helpers.functional as F
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.path.join(
    Path(__file__).parent,
    "data_examples"
)

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Agents & Data Initialization
    # -----------------------------------------------------------------
    F.print_title("1.0 Starting the Profile Auditor App")

    F.print_subtitle("Wake up Visual Agent")

    try:
        print("\t>> Freelancer Profile Auditor - Visual Brand (Vision Model)")
        visual_brand_evaluator = VisualBrandEvaluator(
            model_name          = CFG.GEMINI_FLASH, 
            system_prompt       = VISUAL_BRAND_PROMPT,
            structured_response = VisualBrandEvaluationSchema,
        )

        F.print_success_message("Visual Agent Loaded Successfully")
    
    except Exception as e:
        F.print_error_message("Error While Loading Visual Agent")
        F.print_error_message(e)
        exit()

    F.print_subtitle("Loading Profile Data")
    
    try:
        print("\t>> Loading Profile Image Samples")
        profile_data_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "visual_brand_samples.json")
        )

        F.print_success_message("Data Loaded Successfully")
    
    except Exception as e:
        F.print_error_message("Error While Loading Data")
        F.print_error_message(e)
        exit()

    # ==================================================================
    # 2.0 Testing Agents
    # ==================================================================
    for profile_sample in profile_data_samples:
        freelancer_name = profile_sample["freelancer_name"]
        job_role = profile_sample["job_role"]  
        img_path = os.path.join(DATA_PATH, profile_sample["image_name"])

        print(f"\t>> Directly Analyzing Profile Image for: {freelancer_name}")
        print(f"\t>> Role Context: {job_role}")
        
        visual_brand_result = visual_brand_evaluator.invoke(
            image_path = img_path, 
            job_role = job_role
        )
        
        F.print_structured_response(visual_brand_result)
        print("-" * 50)