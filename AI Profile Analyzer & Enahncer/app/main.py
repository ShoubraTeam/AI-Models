# ---------------------------------------------------------------------
# The Main Workflow - Freelancer Profile Auditor (Visual, Bio & Skills)
# ---------------------------------------------------------------------

# agents
from agents import VisualBrandEvaluator, BioAnalyzer, SkillsAnalyzer

# schemas
from schemas import VisualBrandEvaluationSchema, BioAnalyzerSchema, SkillsAnalyzerSchema

# prompts
from prompts import VISUAL_BRAND_PROMPT, BIO_ANALYZER_PROMPT, SKILLS_ANALYZER_PROMPT

# others
import os
import time
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

    F.print_subtitle("Wake up Profile Auditor Agents")

    try:
        print("\t>> Freelancer Profile Auditor - Visual Brand (Vision Model)")
        visual_brand_evaluator = VisualBrandEvaluator(
            model_name          = CFG.GEMINI_FLASH, 
            system_prompt       = VISUAL_BRAND_PROMPT,
            structured_response = VisualBrandEvaluationSchema,
        )

        print("\t>> Freelancer Profile Auditor - Bio Copywriting (LLM Model)")
        bio_analyzer = BioAnalyzer(
            model_name          = CFG.GROQ_LLAMA_70b,
            system_prompt       = BIO_ANALYZER_PROMPT,
            structured_response = BioAnalyzerSchema,
        )

        print("\t>> Freelancer Profile Auditor - Skills & Domain Alignment (LLM Model)")
        skills_analyzer = SkillsAnalyzer(
            model_name          = CFG.GROQ_LLAMA_8b, 
            system_prompt       = SKILLS_ANALYZER_PROMPT,
            structured_response = SkillsAnalyzerSchema,
        )

        F.print_success_message("All Profile Agents Loaded Successfully")
    
    except Exception as e:
        F.print_error_message("Error While Loading Profile Agents")
        F.print_error_message(e)
        exit()

    F.print_subtitle("Loading Profile Data")
    
    try:
        print("\t>> Loading Profile Image Samples")
        visual_brand_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "visual_brand_samples.json")
        )

        print("\t>> Loading Profile Bio Samples")
        bio_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "bio_samples.json")
        )

        print("\t>> Loading Profile Skills Samples")
        skills_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "skills_samples.json")
        )

        F.print_success_message("All Data Samples Loaded Successfully")
    
    except Exception as e:
        F.print_error_message("Error While Loading Data Samples")
        F.print_error_message(e)
        exit()

    # ==================================================================
    # 2.0 Testing Agents
    # ==================================================================
    F.print_title("2.0 Testing Profile Agents")

    # --------------------------------------------
    # Pipeline 1: Visual Brand Assessment
    # --------------------------------------------
    F.print_subtitle("Sub-Agent 1: Freelancer Profile Visual Brand Audit")

    for idx, profile_sample in enumerate(visual_brand_samples):

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


    # --------------------------------------------
    # Pipeline 2: Bio Copywriting Assessment
    # --------------------------------------------
    F.print_subtitle("Sub-Agent 2: Freelancer Profile Bio / Summary Audit")

    for idx, bio_sample in enumerate(bio_samples):
        
        freelancer_name = bio_sample["freelancer_name"]
        job_role = bio_sample["job_role"]  
        bio_text = bio_sample["bio_text"]

        print(f"\t>> Critically Analyzing Copywriting Bio for: {freelancer_name}")
        print(f"\t>> Role Context: {job_role}")
        
        bio_result = bio_analyzer.invoke(
            bio_text = bio_text, 
            job_role = job_role
        )
        
        F.print_structured_response(bio_result)
        print("-" * 50)


    # --------------------------------------------
    # Pipeline 3: Skills & Domain Alignment Assessment
    # --------------------------------------------
    F.print_subtitle("Sub-Agent 3: Freelancer Profile Skills Alignment Audit")

    for idx, skills_sample in enumerate(skills_samples):

        freelancer_name = skills_sample["freelancer_name"]
        job_role = skills_sample["job_role"]  
        declared_skills = skills_sample["declared_skills"] 

        print(f"\t>> Verifying Technical Skills Alignment for: {freelancer_name}")
        print(f"\t>> Role Context: {job_role}")
        
        skills_result = skills_analyzer.invoke(
            declared_skills = declared_skills, 
            job_role = job_role
        )
        
        F.print_structured_response(skills_result)
        print("-" * 50)