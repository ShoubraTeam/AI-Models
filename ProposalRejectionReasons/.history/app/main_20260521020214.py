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
We need a corporate blog website. 
The project must be built STRICTLY using Python and Django. No PHP or WordPress allowed.
Also, the project must be fully delivered within 5 days due to a strict marketing launch deadline.
"""

    proposal_text = """
Hi, I will build your web app using React and Node.js. 
For security, I will implement OAuth2 with JWT for secure user sessions. 
I will also configure cron jobs on AWS to trigger daily database snapshots to S3 at 00:00 UTC. 
To secure data in transit, I will enforce HTTPS with SSL/TLS encryption.
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