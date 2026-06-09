import ast
import json
import re
from typing import Any


def find_failed_generation(value: Any) -> str | None:
    if isinstance(value, dict):
        if "failed_generation" in value:
            return value["failed_generation"]

        for item in value.values():
            found = find_failed_generation(item)
            if found is not None:
                return found

    elif isinstance(value, (list, tuple)):
        for item in value:
            found = find_failed_generation(item)
            if found is not None:
                return found

    return None


def extract_failed_generation(error: Exception) -> str | None:
    for attr_name in ("body", "response", "args"):
        attr_value = getattr(error, attr_name, None)
        found = find_failed_generation(attr_value)
        if found is not None:
            return found

    error_text = str(error)
    match = re.search(
        r"['\"]failed_generation['\"]\s*:\s*('(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\")",
        error_text,
        flags=re.DOTALL,
    )
    if not match:
        return None

    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return match.group(1).strip("'\"")


def extract_json_payload(raw_generation: str) -> Any:
    raw_generation = raw_generation.strip()

    if raw_generation.startswith("<function="):
        raw_generation = raw_generation.split(">", 1)[1]
        raw_generation = raw_generation.rsplit("</function>", 1)[0]

    start_idx = raw_generation.find("{")
    end_idx = raw_generation.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("Failed generation did not contain a JSON object.")

    payload = json.loads(raw_generation[start_idx:end_idx + 1])

    if isinstance(payload, dict) and "arguments" in payload:
        payload = payload["arguments"]
        if isinstance(payload, str):
            payload = json.loads(payload)

    return payload


def recover_structured_response(error: Exception, structured_response):
    if structured_response is None:
        raise error

    failed_generation = extract_failed_generation(error)
    if not failed_generation:
        raise error

    payload = extract_json_payload(failed_generation)
    return structured_response.model_validate(payload)
