# ---------------------------------------------------------------------
# The Evaluation Workflow
# ---------------------------------------------------------------------

# agents
from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobRequirementsExtractor, JobRequirementsMatcher
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator

# schemas
from schemas import JobToolResponse, ProposalToolsResponse
from schemas import JobKeyPointsSchema, JobUnderstandingEvalSchema
from schemas import ExtractedRequirementsSchema, RequirementCoverageSchema

# prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT, PROPOSAL_TOOLS_EXTRACTION_PROMPT
from prompts import JOB_KEY_POINTS_EXTRACTION_PROMPT, JOB_UNDERSTANDING_EVALUATOR_PROMPT
from prompts import REQUIREMENT_EXTRACTOR_PROMPT, REQUIREMENT_MATCHER_PROMPT

# data processing
from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer, calc_tools_alignment_score
from processing.job_understanding_processing import calc_job_understanding_result

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
    args = F.parse_terminal_arguments()

    for k, v in args.items():
        print(f"{k} => {v}")
