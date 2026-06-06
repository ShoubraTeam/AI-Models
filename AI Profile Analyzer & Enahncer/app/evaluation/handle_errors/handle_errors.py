import ast
import json
import re
from groq import BadRequestError

import traceback
from pathlib import Path


def extract_tool_call_payload(failed_generation: str) -> tuple[str | None, dict | None]:
    """
    Extracts:
    <function=ProposalToolsResponse> {...}</function>

    Returns:
        function_name, parsed_json
    """

    match = re.search(
        r"<function=([^>]+)>\s*(\{.*\})\s*</function>",
        failed_generation,
        re.DOTALL
    )

    if not match:
        return None, None

    function_name = match.group(1)
    payload_str = match.group(2)

    # Convert Python-like literals to JSON-compatible literals
    payload_str = (
        payload_str
        .replace(": True", ": true")
        .replace(": False", ": false")
        .replace(": None", ": null")
    )

    try:
        return function_name, json.loads(payload_str)
    except json.JSONDecodeError:
        # Fallback if it is Python-like dict syntax
        try:
            return function_name, ast.literal_eval(match.group(2))
        except Exception:
            return function_name, None



def parse_groq_error(e: BadRequestError) -> dict:
    info = {
        "error_type": type(e).__name__,
        "error_code": None,
        "message": None,
        "failed_generation": None,
        "function_name": None,
        "parsed_generation": None,
        "summary_length": None,
        "tool_reviews_count": None,
        "has_extra_type_field": False,
        "raw_error": str(e),
    }

    body = getattr(e, "body", None)

    if isinstance(body, dict):
        error_data = body.get("error", {})
    else:
        error_data = {}

    # Fallback: parse dict from str(e)
    if not error_data:
        raw = str(e)

        try:
            dict_part = raw[raw.index("{"):]
            parsed_raw = ast.literal_eval(dict_part)
            error_data = parsed_raw.get("error", {})
        except Exception:
            error_data = {}

    info["message"] = error_data.get("message")
    info["error_code"] = error_data.get("code")
    failed_generation = error_data.get("failed_generation")
    info["failed_generation"] = failed_generation

    if not failed_generation:
        return info

    function_name, parsed_generation = extract_tool_call_payload(failed_generation)

    info["function_name"] = function_name
    info["parsed_generation"] = parsed_generation

    if not isinstance(parsed_generation, dict):
        return info

    if "summary" in parsed_generation:
        info["summary_length"] = len(parsed_generation["summary"])

    if "tool_reviews" in parsed_generation:
        info["tool_reviews_count"] = len(parsed_generation["tool_reviews"])

    if "tools" in parsed_generation:
        info["tools_count"] = len(parsed_generation["tools"])

    if "type" in parsed_generation:
        info["has_extra_type_field"] = True

    return info




def get_short_error_info(error: Exception) -> dict[str, str | int]:
    tb = traceback.extract_tb(error.__traceback__)
    last_call = tb[-1]

    return {
        "error_type": type(error).__name__,
        "file"      : Path(last_call.filename).name,
        "line"      : last_call.lineno,
        "function"  : last_call.name,
        "message"   : str(error),
    }