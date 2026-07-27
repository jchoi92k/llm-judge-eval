# tests/test_prompt_schema.py

"""
Tests for build_output_schema_from_rubric in evaluation_pipeline.prompts.

Uses a synthetic rubric matching the real rubric JSON structure:
{"rubrics": [{"name": ..., "criteria": [{"criterion": ..., "ratings": {"1": ..., ...}}]}]}
"""

import json
import re

import pytest

from evaluation_pipeline.prompts import build_output_schema_from_rubric


def make_rubric():
    return {
        "rubrics": [
            {
                "name": "Mathematical Accuracy",
                "criteria": [
                    {"criterion": "Validity", "ratings": {"1": "bad", "2": "ok", "3": "good", "4": "great"}},
                    {"criterion": "Clarity and Labeling", "ratings": {"1": "bad", "2": "ok", "3": "good", "4": "great"}},
                ],
            },
            {
                "name": "Pedagogical Quality",
                "criteria": [
                    {"criterion": "Feedback", "ratings": {"1": "bad", "2": "ok", "3": "good", "4": "great"}},
                ],
            },
            {
                "name": "Equity and Fairness",
                "criteria": [
                    {"criterion": "Feedback tone", "ratings": {"1": "bad", "2": "ok", "3": "good"}},
                ],
            },
        ]
    }


def render_to_valid_json(schema: str) -> dict:
    """Replace the unquoted placeholders with concrete values so the schema parses as JSON."""
    concrete = re.sub(r"<\d+-\d+ or null>", "1", schema)
    concrete = re.sub(r"<\d+-\d+>", "1", concrete)
    concrete = concrete.replace("<true/false>", "true")
    return json.loads(concrete)


class TestBuildOutputSchemaFromRubric:
    def test_structure_is_valid_pseudo_json(self):
        schema = build_output_schema_from_rubric(json.dumps(make_rubric()))
        parsed = render_to_valid_json(schema)
        assert set(parsed.keys()) == {"scores", "explanations", "mathematical_accuracy_relevance"}

    def test_category_keys_are_snake_cased(self):
        schema = build_output_schema_from_rubric(json.dumps(make_rubric()))
        parsed = render_to_valid_json(schema)
        assert set(parsed["scores"].keys()) == {
            "Mathematical_Accuracy",
            "Pedagogical_Quality",
            "Equity_and_Fairness",
        }
        assert set(parsed["explanations"].keys()) == set(parsed["scores"].keys())

    def test_criterion_keys_are_snake_cased(self):
        schema = build_output_schema_from_rubric(json.dumps(make_rubric()))
        parsed = render_to_valid_json(schema)
        assert set(parsed["scores"]["Mathematical_Accuracy"].keys()) == {
            "Validity",
            "Clarity_and_Labeling",
        }
        assert set(parsed["scores"]["Equity_and_Fairness"].keys()) == {"Feedback_tone"}

    def test_score_ranges_come_from_ratings(self):
        schema = build_output_schema_from_rubric(json.dumps(make_rubric()))
        # Equity category has ratings 1-3; others 1-4
        assert '"Feedback_tone": <1-3>' in schema
        assert '"Feedback": <1-4>' in schema

    def test_mathematical_accuracy_allows_null(self):
        schema = build_output_schema_from_rubric(json.dumps(make_rubric()))
        assert '"Validity": <1-4 or null>' in schema
        # Non-math categories must not get the null option
        assert '"Feedback": <1-4 or null>' not in schema

    def test_category_name_with_colon_suffix_is_trimmed(self):
        rubric = make_rubric()
        rubric["rubrics"][0]["name"] = "Mathematical Accuracy: scored 1-4"
        schema = build_output_schema_from_rubric(json.dumps(rubric))
        parsed = render_to_valid_json(schema)
        assert "Mathematical_Accuracy" in parsed["scores"]

    def test_relevance_block_present(self):
        schema = build_output_schema_from_rubric(json.dumps(make_rubric()))
        parsed = render_to_valid_json(schema)
        mar = parsed["mathematical_accuracy_relevance"]
        assert set(mar.keys()) == {
            "applicable",
            "explanation",
            "extracted_mathematical_content",
            "catastrophic_errors",
        }
