# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------


# agents
from agents import JobToolsExtractor
from agents import ProposalToolsAnalyzer

# schemas
from schemas import JobToolResponse, ProposalToolsResponse

# prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT, PROPOSAL_TOOLS_EXTRACTION_PROMPT

# data processing
from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer

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

    proposal_tools_analyzer = ProposalToolsAnalyzer(
        model_name = CFG.GROQ_LLAMA_70b,
        system_prompt = PROPOSAL_TOOLS_EXTRACTION_PROMPT,
        structured_response = ProposalToolsResponse,
        model_provider = CFG.PROVIDER_GROQ,
        temperature = CFG.MODELS_CFG["tools_alignment_pipeline"]["proposal_tools_analyzer_temperature"],
        max_tokens = CFG.MODELS_CFG["tools_alignment_pipeline"]["proposal_tools_analyzer_max_tokens"],
    )


    print("- Loading data...")
    tools_alignment_data_samples = F.load_json(
        file_path = os.path.join(DATA_PATH, "tools_alignment_tools.json")
    )


    F.print_title("2.0 Testing the Agents")
    sample = tools_alignment_data_samples[0]

    job_description = sample["job_desc"]
    proposal = sample["proposal1"]

    print("- Extracting Job Tools...")
    job_tools_response = job_tool_extractor.invoke(
        input = job_description
    )
    F.print_structured_response(job_tools_response)

    print("- Analyzing Proposal Tools...")
    prepared_analysis_tool_ip = format_ip_for_proposal_tools_analyzer(
        job_tools = job_tools_response.tools,
        proposal = proposal
    )


    proposal_tools_analysis = proposal_tools_analyzer.invoke(
        input = prepared_analysis_tool_ip
    )

    F.print_structured_response(proposal_tools_analysis)
    
    


