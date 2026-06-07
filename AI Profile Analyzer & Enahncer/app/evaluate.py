# ---------------------------------------------------------------------
# The Evaluation Workflow
# ---------------------------------------------------------------------



# others
import os
from pathlib import Path
from helpers.evaluation_arguments import get_terminal_arguments
import helpers.functional as F
from dotenv import load_dotenv

# evaluation
from evaluation.evaluate_agents import evaluate_agent_on_task


load_dotenv()

DATA_PATH = os.path.join(
    Path(__file__).parent,
    "data_examples"
)


from pprint import pprint

import logging
if __name__ == "__main__":
    args = get_terminal_arguments()
    evaluate_agent_on_task(
        run_id     = args["run_id"],
        task_name  = args["task"],
        model_name = args["model"],
        **args["model_kwargs"]
    )



