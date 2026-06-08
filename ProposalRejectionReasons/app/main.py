# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()




import os
import time
from pathlib import Path
import logging
import traceback
import helpers.functional as F

from schemas import FinalSubagentResult
from typing import Callable


from superagent_pipeline.proposal_rejection_reasons_pipeline import ProposalsRejectionReasonsPipeline




DATA_PATH = os.path.join(
    Path(__file__).parent.parent,
    "eval_data"
)


EXAMPLES_DATA_PATH = "/mnt/d/Education/College/______GraduationProject/AI-Models/ProposalRejectionReasons/app/data_examples"





def format_error_details(error: Exception) -> str:
    error_lines = []
    current_error = error
    depth = 0

    while current_error is not None:
        prefix = "Root Error" if depth == 0 else f"Caused By #{depth}"
        error_lines.append(f"{prefix}: {type(current_error).__name__}")
        error_lines.append(f"Message: {str(current_error)}")

        if current_error.args:
            error_lines.append(f"Args: {current_error.args}")

        next_error = current_error.__cause__ or current_error.__context__
        if next_error is not None:
            error_lines.append("")

        current_error = next_error
        depth += 1

    traceback_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    error_lines.extend(["", "Traceback:", traceback_text])

    return "\n".join(error_lines)




async def test_full_pipeline():
    # init

    F.print_title("1.0 Starting Full Pipeline Test")
    F.print_subtitle("Initiating Agents")
    start_time = time.time()
    pipeline = ProposalsRejectionReasonsPipeline()
    end_time = time.time()
    print(f">> Init Time: {round(end_time - start_time, 3)} seconds")


    # load data
    F.print_subtitle("Loading Test Data")
    samples = F.load_json(
        os.path.join(EXAMPLES_DATA_PATH, "requirement_coverage_samples.json")
    )

    sample = samples[0]
    job_desc = sample.get("job_desc", None)
    proposal = sample['proposals'][0]

    print("Data Loaded:")
    print(f">> Job Desc: {job_desc}")
    print(f">> Proposal: {proposal}")

    # phase 1: job feature extraction
    F.print_subtitle("Extracting Job Features In Parallel")
    start_time = time.time()
    job_features = await pipeline.extract_job_features(
        job_desc = job_desc
    )
    end_time = time.time()

    F.print_subtitle("Extracted Job Features")
    for feature_name, feature_result in job_features.items():
        print(f"\n>> {feature_name.capitalize()} Result:")
        F.print_data(feature_result)
    print(f"\n>> Job Feature Extraction Time: {round(end_time - start_time, 3)} seconds")

    # phase 2: proposal analysis
    F.print_subtitle("Analyzing Proposal In Parallel")
    start_time = time.time()
    subagent_results = await pipeline.analyze_proposal(
        job_desc = job_desc,
        proposal = proposal,
        job_features = job_features
    )
    end_time = time.time()

    F.print_subtitle("Raw Subagent Results")
    for feature_name, feature_result in subagent_results.items():
        print(f"\n>> {feature_name.capitalize()} Result:")
        F.print_data(feature_result)
    print(f"\n>> Proposal Analysis Time: {round(end_time - start_time, 3)} seconds")


    F.print_subtitle("Parsing Subagent Results For Super Agent")
    parsed_subagent_results = pipeline.parse_subagents_results(
        results = subagent_results
    )
    
    print(f"\n>> Parsed Results: {parsed_subagent_results}")

    F.print_subtitle("Calling Super Agent")
    try:
        start_time = time.time()
        super_agent_report = await pipeline.super_agent.ainvoke(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = parsed_subagent_results
        )
        end_time = time.time()
    except Exception as e:
        error_details = format_error_details(e)
        F.print_error_message("Super Agent Error Details:")
        print(error_details)
        return

    print(">> Super Agent Result:")
    print(pipeline.format_final_result(super_agent_response = super_agent_report))
    print(f"\n>> Super Agents Time: {round(end_time - start_time, 3)} seconds")

    F.print_success_message("Full pipeline test completed")


async def main():
    await test_full_pipeline()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    
    

    
