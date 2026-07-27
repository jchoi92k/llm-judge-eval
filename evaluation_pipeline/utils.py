# evaluation_pipeline/utils.py

"""
Utility functions for the evaluation pipeline.
"""

import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tiktoken

# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_any_tabular_data(row: pd.Series, dataset_name) -> str:
    """
    Format a pandas row into a readable markdown text format for prompts where each column is a separate section.
    
    Args:
        row: Pandas Series representing a data row
        dataset_name: Name of the dataset for header
        
    Returns:
        Formatted string with headers and content
    """
    parts = []
    
    # Header with dataset identification
    parts.append("# Educational Data")
    if dataset_name:
        parts.append(f"**Dataset:** {dataset_name}")
    
    # Convert all columns to readable format
    parts.append("## Content")
    for column_name, value in row.items():

        # Skip None values
        if value is None:
            continue
        
        # Skip empty lists/arrays
        if isinstance(value, (list, np.ndarray)) and len(value) == 0:
            continue
        
        # Skip NaN or empty strings
        if not isinstance(value, (list, np.ndarray)) and (pd.isna(value) or value == ''):
            continue
            
        # Clean up column name for display
        readable_column = column_name.replace('_', ' ').replace('-', ' ').title()
        
        # Handle dictionary or lists as JSON strings
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            value_str = str(value).strip()
        
        # Skip empty values
        if value_str:
            parts.append(f"### {readable_column}\n{value_str}")
    
    return "\n\n".join(parts)

def truncate_text_to_tokens(text: str, max_tokens: int = 8000, model: str = "text-embedding-3-small") -> str:
    """
    Truncate text to fit within token limit for embedding models.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens (default 8000, leaving buffer for 8192 limit)
        model: Model name for tokenizer; defaults to cl100k_base
        
    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

# ============================================================================
# PARSING UTILITIES
# ============================================================================

def extract_json_from_string(text: str) -> str:
    """
    Extract JSON content from a string that might have markdown formatting.
    
    Args:
        text: String potentially containing JSON in markdown code blocks
        
    Returns:
        Extracted JSON string without markdown formatting
    """
    # Try to find JSON in markdown code block
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic code block
    code_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return text.strip()


def try_parse_evaluation(eval_text: str) -> Tuple[bool, Optional[Dict], str]:
    """
    Attempt to parse an evaluation string as JSON with automatic fixes.
    
    Args:
        eval_text: String containing evaluation JSON
        
    Returns:
        Tuple of (success: bool, parsed_data: dict or None, error_message: str)
    """

    # Handle empty strings
    if not eval_text:
        return False, None, "Empty evaluation string"
    
    # Extract potential JSON content
    json_content = extract_json_from_string(eval_text)
    try:
        parsed = json.loads(json_content)
        _coerce_scores_to_numeric(parsed)
        return True, parsed, ""

    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {str(e)}"

        # Try common fixes
        try:
            fixed_content = json_content.replace("'", '"')
            parsed = json.loads(fixed_content)
            _coerce_scores_to_numeric(parsed)
            return True, parsed, "Fixed by replacing single quotes"
        except:
            pass
        try:
            fixed_content = re.sub(r',\s*([}\]])', r'\1', json_content)
            parsed = json.loads(fixed_content)
            _coerce_scores_to_numeric(parsed)
            return True, parsed, "Fixed by removing trailing commas"
        except:
            pass
        return False, None, error_msg


def _coerce_scores_to_numeric(parsed: Dict) -> None:
    """Coerce string scores to int/float in-place. Handles LLMs returning '3' instead of 3."""
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        return
    for category, subcats in scores.items():
        if not isinstance(subcats, dict):
            continue
        for key, val in subcats.items():
            if isinstance(val, str):
                try:
                    subcats[key] = int(val)
                except ValueError:
                    try:
                        subcats[key] = float(val)
                    except ValueError:
                        pass  # leave as-is if not numeric

    # Normalize mathematical_accuracy_relevance.applicable to Python bool
    mar = parsed.get("mathematical_accuracy_relevance")
    if isinstance(mar, dict) and "applicable" in mar:
        val = mar["applicable"]
        if isinstance(val, str):
            mar["applicable"] = val.strip().lower() == "true"

def extract_text_from_prompts(prompt: List[Dict[str, Any]]) -> str:
    """Extract and concatenate all text parts from a prompt"""
    text_parts = []
    for content_part in prompt[0]['content']:
        if content_part.get('type') == 'input_text':
            text_parts.append(content_part['text'])
    return ''.join(text_parts)

# ============================================================================
# ADJUDICATION LOGIC
# ============================================================================

def needs_adjudication(eval1: Dict, eval2: Dict) -> Tuple[bool, str]:
    """Check if two evaluations need adjudication."""
    # Check mathematical_accuracy_relevance flag discrepancy
    # Normalize to Python bool: handles True/False, "true"/"false", "True"/"False"
    def _to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() == "true"
        return val

    flag1 = _to_bool(eval1.get('mathematical_accuracy_relevance', {}).get('applicable'))
    flag2 = _to_bool(eval2.get('mathematical_accuracy_relevance', {}).get('applicable'))
    if flag1 != flag2:
        return True, "Discrepancy in mathematical_accuracy_relevance flags"

    scores1 = eval1.get('scores', {})
    scores2 = eval2.get('scores', {})
    
    # Check for score discrepancies >= 2
    # Categories hardcoded as per rubric schema
    for category in ['Mathematical_Accuracy', 'Pedagogical_Quality', 'Equity_and_Fairness']:
        cat1 = scores1.get(category, {})
        cat2 = scores2.get(category, {})
        
        for subcategory in cat1.keys():
            val1 = cat1.get(subcategory)
            val2 = cat2.get(subcategory, val1)
            
            if val1 is not None and val2 is not None:
                try:
                    if abs(int(val1) - int(val2)) >= 2:
                        return True, f"Score discrepancy >= 2 in {category} for {subcategory}"
                except (ValueError, TypeError):
                    pass

    return False, ""

# ============================================================================
# Multimodal Input Creation
# ============================================================================

def remove_image_markers(text: str) -> str:
    """
    Remove all [Image: N] markers from text.
    
    Args:
        text: Text containing [Image: N] markers where N is an integer
        
    Returns:
        Text with all image markers removed
    """
    return re.sub(r'\[Image:\s*\d+\]', '', text)

def create_input(full_message: str, image_data_base64: Optional[List] = None) -> List[Dict]:
    """
    Create multimodal input structure for API calls with text and images.
    
    Args:
        full_message: Full rendered prompt with [Image: N] markers; image files will be inserted at these markers sequentially
        image_data_base64: List of base64 encoded images (can contain None)
        
    Returns:
        List containing a single dict with role and content parts
    """
    content_parts = []
    
    # Handle NaN or float values from pandas
    if image_data_base64 is None or isinstance(image_data_base64, float) or not any(img is not None for img in image_data_base64):
        # No images, return just the text
        content_parts.append({
            "type": "input_text",
            "text": full_message
        })
        return [{"role": "user", "content": content_parts}]
    
    # Filter out None images and create image list
    images = [img for img in image_data_base64 if img is not None]
    
    # Split message by image markers
    parts = full_message.split('[Image:')
    
    # Add first text part
    if parts[0].strip():
        content_parts.append({
            "type": "input_text",
            "text": parts[0].strip()
        })
    
    # Process remaining parts (each starts with an image number)
    for i, part in enumerate(parts[1:]):
        # Add the corresponding image
        if i < len(images):
            content_parts.append({
                "type": "input_image",
                "image_url": images[i]
            })
        
        # Extract text after the image marker (e.g., "1]\nsome text" -> "some text")
        text_after_image = part.split(']', 1)[-1].strip()
        if text_after_image:
            content_parts.append({
                "type": "input_text",
                "text": text_after_image
            })
    return [{"role": "user", "content": content_parts}]

# ============================================================================
# Other utilities
# ============================================================================

def flush_logs():
    """Ensure all logged output is displayed before interactive prompts."""
    for handler in logging.root.handlers:
        handler.flush()
    sys.stdout.flush()
    print()


def count_images_in_prompt(prompt: List[Dict[str, Any]]) -> int:
    """
    Count the number of images in a prompt structure.

    Args:
        prompt: A prompt as a list of message dicts with content parts.

    Returns:
        Number of image content parts found.
    """
    count = 0
    for message in prompt:
        content = message.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_image":
                    count += 1
    return count


def estimate_image_tokens(n_images: int, tokens_per_image: int = 1100) -> int:
    """
    Estimate the total tokens consumed by images in an API call.

    OpenAI charges image tokens based on resolution and detail level.
    At high detail, a typical image uses ~765-1100 tokens (depending on size).
    Default of 1100 is a conservative estimate for whiteboard/diagram images.

    Args:
        n_images: Number of images in the prompt.
        tokens_per_image: Estimated tokens per image (default 1100).

    Returns:
        Estimated total image tokens.
    """
    return n_images * tokens_per_image


def find_prefix(all_prompts: List[List[Dict[str, Any]]]) -> Tuple[str, str]:
    """
    Find the longest common prefix among all prompts and return it along with the remaining text.

    Args:
        all_prompts: List of prompts, where each prompt is a list of dictionaries with text content.

    Returns:
        A tuple containing the common prefix string and the remaining text string.
    """
    all_texts = [extract_text_from_prompts(prompt) for prompt in all_prompts]

    # Find common prefix among all text versions
    cached_prefix = ""
    if all_texts:
        prefix = []
        for chars in zip(*all_texts):
            if len(set(chars)) == 1:  # All characters are the same
                prefix.append(chars[0])
            else:
                break
        cached_prefix = "".join(prefix)

    # Estimate cost per prompt; estimate using first prompt as sample instead of running tiktoken on all texts
    sample_text = all_texts[0] if all_texts else ""
    uncached_text = sample_text.replace(cached_prefix, '', 1)  # Remove only first occurrence

    return cached_prefix, uncached_text