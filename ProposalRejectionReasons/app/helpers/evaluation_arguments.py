# ---------------------------------------------------------------------------------
# Implementing helpful functions to
# - accept terminal arguments
# - parse configurations
# - return run configurations
# ---------------------------------------------------------------------------------

from helpers.config import EVALUATION_MODELS_MAPPING
from argparse import ArgumentParser


def parse_terminal_arguments():
    """
    Return dict of given arguments
    """

    parser = ArgumentParser(description = "Parsing Terminal Arguments")
    
    parser.add_argument(
        "--run_id", 
        type = int,
        help = "The run ID.",
        required = True
    )


    parser.add_argument(
        "--task", 
        type = str,
        help = "The task to evaluate, tools_alignment, job_understanding, etc...",
        required = True
    )


    parser.add_argument(
        "--model",
        type = str,
        help = "The model used in evaluating.",
        required = True
    )

    parser.add_argument(
        "--temperature",
        type = float,
        help = "The temperature that controls the model's randomness.",
        required = True
    )

    parser.add_argument(
        "--max_tokens",
        type = int,
        help = "The max_tokens that the model should generate.",
        required = True
    )


    parser.add_argument(
        "--top_p",
        type = float,
        help = "The top_p that control's selected tokens",
        required = True
    )


    args = parser.parse_args().__dict__
    return args


def get_terminal_arguments():
    """
    Accepting the terminal arguments & parse them
    """

    args = parse_terminal_arguments()


    # parse model
    model = args["model"]
    if model not in EVALUATION_MODELS_MAPPING:
        raise ValueError(f"Unknown model alias: {model}")
    args["model"] = EVALUATION_MODELS_MAPPING[model]

    # parse model args
    kwargs = {}
    for arg in ["temperature", "max_tokens", "top_p"]:
        value = args.pop(arg, None)

        if value is not None:
            kwargs[arg] = value
    
    args["model_kwargs"] = kwargs

    return args