# ---------------------------------------------------------------------
# The Main Workflow - Integrated Freelancer Profile Auditor (Orchestrated Suite)
# ---------------------------------------------------------------------

# agents
from agents import VisualBrandEvaluator, BioAnalyzer, SkillsAnalyzer, NumericalAnalyzer, SuperAgent

# schemas
from schemas import VisualBrandEvaluationSchema, BioAnalyzerSchema, SkillsAnalyzerSchema, NumericalAnalyzerSchema, SuperAgentSchema

# prompts
from prompts import VISUAL_BRAND_PROMPT, BIO_ANALYZER_PROMPT, SKILLS_ANALYZER_PROMPT, SUPER_AGENT_PROMPT

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
    # 1.0 Agents & Engines Initialization
    # -----------------------------------------------------------------
    F.print_title("1.0 Starting the Profile Auditor Orchestrator")

    F.print_subtitle("Wake up Profile Auditor Agents Suite")

    try:
        print("\t>> Loading Sub-Agent 1: Visual Brand Evaluator (Vision Model)")
        visual_brand_evaluator = VisualBrandEvaluator(
            model_name          = CFG.GEMINI_FLASH_LITE, 
            system_prompt       = VISUAL_BRAND_PROMPT,
            structured_response = VisualBrandEvaluationSchema,
        )

        print("\t>> Loading Sub-Agent 2: Bio Copywriting Analyzer (LLM Model)")
        bio_analyzer = BioAnalyzer(
            model_name          = CFG.GROQ_LLAMA_70b,
            system_prompt       = BIO_ANALYZER_PROMPT,
            structured_response = BioAnalyzerSchema,
        )

        print("\t>> Loading Sub-Agent 3: Skills Alignment Analyzer (LLM Model)")
        skills_analyzer = SkillsAnalyzer(
            model_name          = CFG.GROQ_LLAMA_8b, 
            system_prompt       = SKILLS_ANALYZER_PROMPT,
            structured_response = SkillsAnalyzerSchema,
        )

        print("\t>> Loading Sub-Engine 4: Numerical Metrics Engine (Pure Python Rule-Based)")
        numerical_analyzer = NumericalAnalyzer(
            model_name          = "deterministic_python",
            system_prompt       = "none",
            structured_response = NumericalAnalyzerSchema
        )

        print("\t>> Loading THE BOSS: Master Orchestrator (SuperAgent LLM)")
        super_agent = SuperAgent(
            model_name          = CFG.GROQ_LLAMA_70b, 
            system_prompt       = SUPER_AGENT_PROMPT,
            structured_response = SuperAgentSchema
        )

        F.print_success_message("All Profile Core Engines & SuperAgent Loaded Successfully")
    
    except Exception as e:
        F.print_error_message("Error While Loading Core Orchestration Components")
        F.print_error_message(e)
        exit()

    # -----------------------------------------------------------------
    # Loading Unified Profile Data
    # -----------------------------------------------------------------
    F.print_subtitle("Loading Unified Freelancer Dataset")
    
    try:
        profile_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "profile_samples.json")
        )
        F.print_success_message(f"Successfully Loaded {len(profile_samples)} Unified Freelancer Profiles")
    
    except Exception as e:
        F.print_error_message("Error While Loading Unified profile_samples.json")
        F.print_error_message(e)
        exit()

    # ==================================================================
    # 2.0 Executing Orchestrated Profile Audits
    # ==================================================================
    F.print_title("2.0 Running End-to-End Orchestrated Profile Audits")

    for idx, profile in enumerate(profile_samples):
        freelancer_name = profile["freelancer_name"]
        job_role = profile["job_role"]
        
        print("\n" + "="*70)
        print(f"STARTING FULL AUDIT FOR FREELANCER [{idx + 1}/{len(profile_samples)}]: {freelancer_name}")
        print(f"Target Core Market Domain: {job_role}")
        print("="*70)

        # --------------------------------------------
        # Step 2.1: Visual Brand Assessment
        # --------------------------------------------
        print("\n\t[Step 1/5] Executing Visual Brand Audit...")
        img_path = os.path.join(DATA_PATH, profile["image_name"])
        visual_res = visual_brand_evaluator.invoke(image_path=img_path, job_role=job_role)
        time.sleep(3) # تهدئة للـ Rate Limits بين الموديلات

        # --------------------------------------------
        # Step 2.2: Bio Copywriting Assessment
        # --------------------------------------------
        print("\t[Step 2/5] Executing Bio Copywriting Analysis...")
        bio_res = bio_analyzer.invoke(bio_text=profile["bio_text"], job_role=job_role)
        time.sleep(3)

        # --------------------------------------------
        # Step 2.3: Skills & Domain Alignment Assessment
        # --------------------------------------------
        print("\t[Step 3/5] Checking Technical Skills Alignment...")
        skills_res = skills_analyzer.invoke(declared_skills=profile["declared_skills"], job_role=job_role)
        time.sleep(3)

        # --------------------------------------------
        # Step 2.4: Numerical Metrics Engine Execution
        # --------------------------------------------
        print("\t[Step 4/5] Crunching Numerical and Pricing Performance Metrics...")
        numerical_res = numerical_analyzer.invoke(
            job_role             = job_role,
            hourly_rate          = profile["hourly_rate"],
            rating               = profile["rating"],
            total_completed_jobs = profile["total_completed_jobs"]
        )
        # مفيش sleep هنا لأن المحرك بايثون سريع ولحظي!

        # --------------------------------------------
        # Step 2.5: The SuperAgent Consolidation & Synthesis
        # --------------------------------------------
        print("\n\t[Step 5/5] Triggering Master Orchestrator (SuperAgent) for Cross-Domain Synthesis...")
        print("\t>> Ingesting sub-audits outputs to construct the final report...")
        
        final_executive_report = super_agent.invoke(
            visual_res    = visual_res,
            bio_res       = bio_res,
            skills_res    = skills_res,
            numerical_res = numerical_res
        )
        
        # --------------------------------------------
        # Print Final Beautiful Response
        # --------------------------------------------
        F.print_success_message(f"FINAL COHESIVE EXECUTIVE REPORT COMPILED FOR: {freelancer_name}")
        F.print_structured_response(final_executive_report)
        print("\n" + "#"*70 + "\n")
        
        # تهدئة عامة قبل الانتقال للفريلانسر القادم في اللفة الجديدة
        if idx < len(profile_samples) - 1:
            print("Cooling down for 5 seconds before moving to the next profile sample...")
            time.sleep(5)