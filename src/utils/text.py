"""Text processing utilities for VLM response parsing."""

import ast
import json
import logging
import re
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def truncate_repetitive_text(text: str, max_repeat: int = 5) -> str:
    """Truncate consecutive repeated characters to at most max_repeat.

    Handles Chinese, English, and symbol characters.

    Args:
        text: Input text.
        max_repeat: Maximum allowed consecutive repetitions.

    Returns:
        Text with repetitions truncated.
    """
    if not isinstance(text, str):
        return text

    def replace_match(match: re.Match) -> str:
        char = match.group(1)
        return char * min(len(match.group(0)), max_repeat)

    result = re.sub(r"(.)\1*", replace_match, text)
    return result


def parse_vlm_response(
    vlm_response: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Parse a VLM response into a validated dictionary.

    Handles various formats: dict, JSON string, JSON with extra text,
    and markdown code blocks.

    Args:
        vlm_response: Raw VLM response (string or dict).

    Returns:
        Validated response dictionary.

    Raises:
        ValueError: If parsing or validation fails.
    """
    # If already a dict, validate and return
    if isinstance(vlm_response, dict):
        vlm_response = _validate_vlm_response(vlm_response)
        return vlm_response

    # Convert to string if not already
    if not isinstance(vlm_response, str):
        vlm_response = str(vlm_response)

    # Try to extract JSON from markdown code blocks
    code_block_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```", vlm_response, re.DOTALL
    )
    if code_block_match:
        code_content = code_block_match.group(1).strip()
        # Find JSON object within the code block
        first_brace = code_content.find("{")
        last_brace = code_content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            vlm_response = code_content[first_brace : last_brace + 1]

    # Try to find JSON object in the string if not already extracted
    if not vlm_response.strip().startswith("{"):
        first_brace = vlm_response.find("{")
        last_brace = vlm_response.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            vlm_response = vlm_response[first_brace : last_brace + 1]

    # Try to fix common bracket mismatches by counting braces
    open_braces = vlm_response.count("{")
    close_braces = vlm_response.count("}")
    if open_braces > close_braces:
        vlm_response += "}" * (open_braces - close_braces)
    elif close_braces > open_braces:
        vlm_response = "{" * (close_braces - open_braces) + vlm_response

    # Try direct JSON parsing
    try:
        result = json.loads(vlm_response)
        result = _validate_vlm_response(result)
        return result
    except json.JSONDecodeError:
        pass

    # Try replacing single quotes with double quotes (heuristic)
    try:
        fixed_response = re.sub(r"'(\w+)':", r'"\1":', vlm_response)
        fixed_response = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_response)
        result = json.loads(fixed_response)
        result = _validate_vlm_response(result)
        return result
    except (json.JSONDecodeError, Exception):
        pass

    # Last resort: try ast.literal_eval for Python dict syntax
    try:
        result = ast.literal_eval(vlm_response)
        if isinstance(result, dict):
            result = _validate_vlm_response(result)
            return result
    except (ValueError, SyntaxError):
        pass

    # If all parsing attempts fail, raise an error with context
    logger.error(
        "Failed to parse VLM response: %s...", vlm_response[:200]
    )
    raise ValueError(
        f"Failed to parse VLM response. Response preview: {vlm_response}..."
    )


def _validate_vlm_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that the VLM response contains required fields.

    For local window: requires 'rationale' and 'boxes'.
    For global window: requires 'rationale' and 'persons'.

    Args:
        response: Parsed VLM response dictionary.

    Returns:
        Validated (and possibly normalized) response dictionary.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    if not isinstance(response, dict):
        raise ValueError(f"VLM response must be a dict, got {type(response)}")

    if "rationale" not in response:
        raise ValueError("VLM response must contain 'rationale' field")

    # Check for either 'boxes' (local window) or 'persons' (global window)
    if "boxes" not in response and "persons" not in response:
        raise ValueError(
            "VLM response must contain either 'boxes' or 'persons' field"
        )

    # Validate boxes structure if present
    if "boxes" in response:
        if not isinstance(response["boxes"], list):
            raise ValueError("'boxes' field must be a list")

        for i, box in enumerate(response["boxes"]):
            if isinstance(box, list):
                response["boxes"][i] = {"frame_id": i, "bbox": box}
            elif not isinstance(box, dict):
                raise ValueError("Each box must be a dict or a bbox list")

        for box in response["boxes"]:
            if "frame_id" not in box or "bbox" not in box:
                raise ValueError(
                    "Each box must contain 'frame_id' and 'bbox' fields"
                )

    # Validate persons structure if present
    if "persons" in response:
        if not isinstance(response["persons"], dict):
            raise ValueError("'persons' field must be a dict")

    return response
