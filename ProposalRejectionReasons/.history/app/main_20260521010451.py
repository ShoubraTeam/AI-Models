# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------

# agents
from agents import JobToolsExtractor
from agents.requirement_coverage.requirement_coverage_agent import RequirementCoverageAgent

# schemas
from schemas import JobToolResponse

# prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT

# others
import helpers.config as CFG
import helpers.functional as F
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    F.print_title("1.0 Starting the APP")
    
    print("- Waking Up Agents...")

    job_tool_extractor = JobToolsExtractor(
        model_name = CFG.GROQ_LLAMA_8b,
        system_prompt = JOB_TOOLS_EXTRACTION_PROMPT,
        structured_response = JobToolResponse,
        model_provider = CFG.PROVIDER_GROQ,
        temperature = CFG.MODELS_CFG["tools_alignment_pipeline"]["job_tools_extractor_temperature"],
        max_tokens = CFG.MODELS_CFG["tools_alignment_pipeline"]["job_tools_extractor_max_tokens"],
    )

    requirement_agent = RequirementCoverageAgent(
        extractor_model=CFG.GROQ_LLAMA_70b,
        matcher_model=CFG.GROQ_LLAMA_70b
    )

    print("- Loading data...")
    job_description = """
We need a web-based Inventory Management System for our warehouse. 
The core and mandatory feature is that warehouse workers must be able to bulk-upload product data using Excel/CSV files to update the stock automatically. 

Important Note: This is Phase 1. Do NOT implement any payment gateways or online checkout features now; we strictly prohibit online transactions in this version to avoid security compliance issues. 

The system must include a clean dashboard showing low-stock alerts.
"""

    proposal_text = """
Dear Client, 
I am a senior full-stack web developer with over 7 years of experience in building modern websites. 
I have worked on many projects similar to yours and I always deliver high-quality work on time and within budget. 
Please check my profile and portfolio to see my previous work. 
I am available to start immediately and would love to jump on a quick call to discuss how I can help you. 
Thank you for your time!
"""

    F.print_title("2.0 Testing Tool Agent")
    tool_response = job_tool_extractor.invoke(
        input = job_description
    )
    F.print_title("3.0 Printing Tool Output")
    F.print_structured_response(tool_response)

    F.print_title("4.0 Testing Requirement Coverage Agent")
    req_response = requirement_agent.invoke(
        job_description = job_description,
        proposal_text = proposal_text
    )

    F.print_title("5.0 Printing Requirement Output")
    F.print_structured_response(req_response)