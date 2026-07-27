# tests/test_utils.py

"""
Tests for pure utility functions in evaluation_pipeline.utils.

All fixtures are synthetic. No API calls, no file IO.
"""

import json

import pytest

from evaluation_pipeline import utils


# ============================================================================
# extract_json_from_string
# ============================================================================

class TestExtractJsonFromString:
    def test_json_code_block(self):
        text = 'Here is the result:\n```json\n{"a": 1}\n```\nDone.'
        assert utils.extract_json_from_string(text) == '{"a": 1}'

    def test_generic_code_block(self):
        text = '```\n{"a": 1}\n```'
        assert utils.extract_json_from_string(text) == '{"a": 1}'

    def test_plain_text_passthrough(self):
        assert utils.extract_json_from_string('  {"a": 1}  ') == '{"a": 1}'


# ============================================================================
# try_parse_evaluation
# ============================================================================

def make_evaluation(score="3", applicable=True):
    """Synthetic evaluation payload matching the pipeline's output schema."""
    return {
        "scores": {
            "Mathematical_Accuracy": {"Validity": score},
            "Pedagogical_Quality": {"Feedback": 2},
        },
        "explanations": {
            "Mathematical_Accuracy": {"Validity": "explanation text"},
        },
        "mathematical_accuracy_relevance": {
            "applicable": applicable,
            "explanation": "some analysis",
        },
    }


class TestTryParseEvaluation:
    def test_empty_string(self):
        success, parsed, msg = utils.try_parse_evaluation("")
        assert success is False
        assert parsed is None
        assert msg == "Empty evaluation string"

    def test_valid_json(self):
        success, parsed, msg = utils.try_parse_evaluation(json.dumps(make_evaluation()))
        assert success is True
        assert msg == ""

    def test_markdown_wrapped_json(self):
        text = "```json\n" + json.dumps(make_evaluation()) + "\n```"
        success, parsed, _ = utils.try_parse_evaluation(text)
        assert success is True
        assert parsed["scores"]["Pedagogical_Quality"]["Feedback"] == 2

    def test_string_scores_coerced_to_int(self):
        success, parsed, _ = utils.try_parse_evaluation(json.dumps(make_evaluation(score="3")))
        assert success is True
        assert parsed["scores"]["Mathematical_Accuracy"]["Validity"] == 3
        assert isinstance(parsed["scores"]["Mathematical_Accuracy"]["Validity"], int)

    def test_string_applicable_coerced_to_bool(self):
        success, parsed, _ = utils.try_parse_evaluation(
            json.dumps(make_evaluation(applicable="True"))
        )
        assert success is True
        assert parsed["mathematical_accuracy_relevance"]["applicable"] is True

    def test_single_quote_fix(self):
        success, parsed, msg = utils.try_parse_evaluation('{"a": "b", "c": 1}'.replace('"', "'"))
        assert success is True
        assert parsed == {"a": "b", "c": 1}
        assert "single quotes" in msg

    def test_trailing_comma_fix(self):
        success, parsed, msg = utils.try_parse_evaluation('{"a": 1, "b": [1, 2,],}')
        assert success is True
        assert parsed == {"a": 1, "b": [1, 2]}
        assert "trailing commas" in msg

    def test_unparseable_text(self):
        success, parsed, msg = utils.try_parse_evaluation("not json at all {{{")
        assert success is False
        assert parsed is None
        assert "JSON decode error" in msg


# ============================================================================
# _coerce_scores_to_numeric
# ============================================================================

class TestCoerceScoresToNumeric:
    def test_int_and_float_strings(self):
        parsed = {"scores": {"Cat": {"a": "3", "b": "3.5", "c": 2, "d": None}}}
        utils._coerce_scores_to_numeric(parsed)
        assert parsed["scores"]["Cat"] == {"a": 3, "b": 3.5, "c": 2, "d": None}

    def test_non_numeric_string_left_alone(self):
        parsed = {"scores": {"Cat": {"a": "N/A"}}}
        utils._coerce_scores_to_numeric(parsed)
        assert parsed["scores"]["Cat"]["a"] == "N/A"

    def test_missing_scores_key_is_safe(self):
        parsed = {"explanations": {}}
        utils._coerce_scores_to_numeric(parsed)  # must not raise

    def test_non_dict_scores_is_safe(self):
        utils._coerce_scores_to_numeric({"scores": [1, 2, 3]})  # must not raise

    def test_applicable_false_string(self):
        parsed = {
            "scores": {"Cat": {"a": 1}},
            "mathematical_accuracy_relevance": {"applicable": "false"},
        }
        utils._coerce_scores_to_numeric(parsed)
        assert parsed["mathematical_accuracy_relevance"]["applicable"] is False

    def test_known_quirk_applicable_not_normalized_without_scores(self):
        # Current behavior: the early return on a missing/non-dict "scores" key also
        # skips the "applicable" flag normalization. Low impact in practice (real
        # evaluations always include scores, and needs_adjudication re-normalizes
        # flags itself), but documented here so a future fix is a conscious choice.
        parsed = {"mathematical_accuracy_relevance": {"applicable": "false"}}
        utils._coerce_scores_to_numeric(parsed)
        assert parsed["mathematical_accuracy_relevance"]["applicable"] == "false"


# ============================================================================
# needs_adjudication
# ============================================================================

def make_scored_eval(validity=3, applicable=True):
    return {
        "scores": {
            "Mathematical_Accuracy": {"Validity": validity},
            "Pedagogical_Quality": {"Feedback": 2},
            "Equity_and_Fairness": {"Feedback_tone": 2},
        },
        "mathematical_accuracy_relevance": {"applicable": applicable},
    }


class TestNeedsAdjudication:
    def test_identical_evals_no_adjudication(self):
        needed, reason = utils.needs_adjudication(make_scored_eval(), make_scored_eval())
        assert needed is False
        assert reason == ""

    def test_gap_of_two_triggers(self):
        needed, reason = utils.needs_adjudication(
            make_scored_eval(validity=1), make_scored_eval(validity=3)
        )
        assert needed is True
        assert "Score discrepancy" in reason

    def test_gap_of_one_does_not_trigger(self):
        needed, _ = utils.needs_adjudication(
            make_scored_eval(validity=2), make_scored_eval(validity=3)
        )
        assert needed is False

    def test_applicability_flag_mismatch_triggers(self):
        needed, reason = utils.needs_adjudication(
            make_scored_eval(applicable=True), make_scored_eval(applicable=False)
        )
        assert needed is True
        assert "mathematical_accuracy_relevance" in reason

    def test_string_flags_normalized_before_comparison(self):
        # "True" (string) vs True (bool) must NOT trigger adjudication
        needed, _ = utils.needs_adjudication(
            make_scored_eval(applicable="True"), make_scored_eval(applicable=True)
        )
        assert needed is False

    def test_string_scores_compared_numerically(self):
        needed, _ = utils.needs_adjudication(
            make_scored_eval(validity="1"), make_scored_eval(validity="4")
        )
        assert needed is True

    def test_none_scores_skipped(self):
        needed, _ = utils.needs_adjudication(
            make_scored_eval(validity=None), make_scored_eval(validity=4)
        )
        assert needed is False

    def test_missing_subcategory_in_second_eval_uses_first_value(self):
        eval2 = make_scored_eval()
        del eval2["scores"]["Mathematical_Accuracy"]["Validity"]
        needed, _ = utils.needs_adjudication(make_scored_eval(validity=1), eval2)
        assert needed is False


# ============================================================================
# Image marker / multimodal input handling
# ============================================================================

class TestRemoveImageMarkers:
    def test_removes_markers(self):
        assert utils.remove_image_markers("a [Image: 1] b [Image: 22] c") == "a  b  c"

    def test_no_markers_unchanged(self):
        assert utils.remove_image_markers("plain text") == "plain text"


class TestCreateInput:
    def test_text_only_when_no_images(self):
        result = utils.create_input("hello world", None)
        assert result == [
            {"role": "user", "content": [{"type": "input_text", "text": "hello world"}]}
        ]

    def test_nan_float_treated_as_no_images(self):
        result = utils.create_input("hello", float("nan"))
        assert result[0]["content"][0]["type"] == "input_text"

    def test_all_none_images_treated_as_no_images(self):
        result = utils.create_input("hello", [None, None])
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "input_text"

    def test_images_inserted_at_markers_in_order(self):
        message = "Intro [Image: 1] middle [Image: 2] end"
        result = utils.create_input(message, ["b64_first", "b64_second"])
        parts = result[0]["content"]
        types = [p["type"] for p in parts]
        assert types == [
            "input_text", "input_image", "input_text", "input_image", "input_text"
        ]
        assert parts[1]["image_url"] == "b64_first"
        assert parts[3]["image_url"] == "b64_second"
        assert parts[0]["text"] == "Intro"
        assert parts[2]["text"] == "middle"
        assert parts[4]["text"] == "end"

    def test_none_entries_filtered_from_image_list(self):
        message = "A [Image: 1] B"
        result = utils.create_input(message, [None, "b64_only"])
        images = [p for p in result[0]["content"] if p["type"] == "input_image"]
        assert len(images) == 1
        assert images[0]["image_url"] == "b64_only"


class TestCountImagesInPrompt:
    def test_counts_only_image_parts(self):
        prompt = utils.create_input("x [Image: 1] y [Image: 2] z", ["a", "b"])
        assert utils.count_images_in_prompt(prompt) == 2

    def test_zero_for_text_only(self):
        prompt = utils.create_input("no images here", None)
        assert utils.count_images_in_prompt(prompt) == 0


class TestEstimateImageTokens:
    def test_default_rate(self):
        assert utils.estimate_image_tokens(3) == 3300

    def test_custom_rate(self):
        assert utils.estimate_image_tokens(2, tokens_per_image=765) == 1530

    def test_zero_images(self):
        assert utils.estimate_image_tokens(0) == 0


# ============================================================================
# Prompt text extraction / common prefix
# ============================================================================

def make_text_prompt(text):
    return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]


class TestExtractTextFromPrompts:
    def test_concatenates_text_parts(self):
        prompt = utils.create_input("A [Image: 1] B", ["img"])
        assert utils.extract_text_from_prompts(prompt) == "AB"

    def test_plain_prompt(self):
        assert utils.extract_text_from_prompts(make_text_prompt("hello")) == "hello"


class TestFindPrefix:
    def test_common_prefix_and_remainder(self):
        prompts = [
            make_text_prompt("SHARED PREAMBLE unique-one"),
            make_text_prompt("SHARED PREAMBLE unique-two"),
        ]
        prefix, uncached = utils.find_prefix(prompts)
        assert prefix == "SHARED PREAMBLE unique-"
        assert uncached == "one"

    def test_no_common_prefix(self):
        prompts = [make_text_prompt("abc"), make_text_prompt("xyz")]
        prefix, uncached = utils.find_prefix(prompts)
        assert prefix == ""
        assert uncached == "abc"

    def test_single_prompt_full_prefix(self):
        prompts = [make_text_prompt("only one")]
        prefix, uncached = utils.find_prefix(prompts)
        assert prefix == "only one"
        assert uncached == ""

    def test_empty_list(self):
        prefix, uncached = utils.find_prefix([])
        assert prefix == ""
        assert uncached == ""
