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


class TestBuildJsonSchemaFromRubric:
    """Strict JSON Schema for API-enforced structured outputs (task 9)."""

    def make_schema(self, rubric=None):
        from evaluation_pipeline.prompts import build_json_schema_from_rubric
        return build_json_schema_from_rubric(json.dumps(rubric or make_rubric()))

    def test_top_level_structure(self):
        schema = self.make_schema()
        assert schema["type"] == "object"
        assert set(schema["properties"].keys()) == {
            "scores", "explanations", "mathematical_accuracy_relevance",
        }
        assert set(schema["required"]) == set(schema["properties"].keys())
        assert schema["additionalProperties"] is False

    def test_every_object_node_is_strict(self):
        """OpenAI strict mode: all objects need required=all-props and additionalProperties=False."""
        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert set(node["required"]) == set(node["properties"].keys())
                    assert node["additionalProperties"] is False
                for v in node.values():
                    walk(v)

        walk(self.make_schema())

    def test_score_bounds_from_ratings(self):
        schema = self.make_schema()
        feedback_tone = schema["properties"]["scores"]["properties"]["Equity_and_Fairness"]["properties"]["Feedback_tone"]
        assert feedback_tone == {"type": "integer", "minimum": 1, "maximum": 3}

    def test_math_accuracy_scores_nullable(self):
        schema = self.make_schema()
        validity = schema["properties"]["scores"]["properties"]["Mathematical_Accuracy"]["properties"]["Validity"]
        assert validity["type"] == ["integer", "null"]
        assert validity["minimum"] == 1
        assert validity["maximum"] == 4

    def test_non_math_scores_not_nullable(self):
        schema = self.make_schema()
        feedback = schema["properties"]["scores"]["properties"]["Pedagogical_Quality"]["properties"]["Feedback"]
        assert feedback["type"] == "integer"

    def test_keys_match_pseudo_schema(self):
        """Both schema builders must agree on category/criterion keys, since
        the prompt text and the API enforcement describe the same output."""
        pseudo = render_to_valid_json(build_output_schema_from_rubric(json.dumps(make_rubric())))
        strict = self.make_schema()
        assert set(strict["properties"]["scores"]["properties"].keys()) == set(pseudo["scores"].keys())
        for cat in pseudo["scores"]:
            assert set(strict["properties"]["scores"]["properties"][cat]["properties"].keys()) == set(pseudo["scores"][cat].keys())

    def test_relevance_block_fields(self):
        schema = self.make_schema()
        mar = schema["properties"]["mathematical_accuracy_relevance"]
        assert set(mar["properties"].keys()) == {
            "applicable", "explanation", "extracted_mathematical_content", "catastrophic_errors",
        }
        assert mar["properties"]["applicable"]["type"] == "boolean"

    def test_a_valid_payload_passes_jsonschema_validation(self):
        """Sanity: a payload shaped like VALID model output validates against the schema."""
        jsonschema = pytest.importorskip("jsonschema")
        schema = self.make_schema()
        payload = {
            "scores": {
                "Mathematical_Accuracy": {"Validity": None, "Clarity_and_Labeling": 4},
                "Pedagogical_Quality": {"Feedback": 3},
                "Equity_and_Fairness": {"Feedback_tone": 2},
            },
            "explanations": {
                "Mathematical_Accuracy": {"Validity": "n/a", "Clarity_and_Labeling": "clear"},
                "Pedagogical_Quality": {"Feedback": "good"},
                "Equity_and_Fairness": {"Feedback_tone": "warm"},
            },
            "mathematical_accuracy_relevance": {
                "applicable": False,
                "explanation": "no math content",
                "extracted_mathematical_content": "",
                "catastrophic_errors": "None",
            },
        }
        jsonschema.validate(payload, schema)


class TestNARatingKeys:
    """Rubrics with a non-numeric 'N/A' rating key (seen in real team rubrics)
    must not crash either builder; such criteria allow null."""

    def make_na_rubric(self):
        rubric = make_rubric()
        rubric["rubrics"][1]["criteria"][0]["ratings"]["N/A"] = "not applicable"
        return rubric

    def test_pseudo_schema_handles_na_key(self):
        schema = build_output_schema_from_rubric(json.dumps(self.make_na_rubric()))
        assert '"Feedback": <1-4 or null>' in schema

    def test_json_schema_handles_na_key(self):
        from evaluation_pipeline.prompts import build_json_schema_from_rubric
        schema = build_json_schema_from_rubric(json.dumps(self.make_na_rubric()))
        feedback = schema["properties"]["scores"]["properties"]["Pedagogical_Quality"]["properties"]["Feedback"]
        assert feedback["type"] == ["integer", "null"]
        assert feedback["minimum"] == 1
        assert feedback["maximum"] == 4

    def test_all_na_criterion_raises(self):
        from evaluation_pipeline.prompts import build_json_schema_from_rubric
        rubric = make_rubric()
        rubric["rubrics"][1]["criteria"][0]["ratings"] = {"N/A": "only"}
        with pytest.raises(ValueError, match="no numeric rating keys"):
            build_json_schema_from_rubric(json.dumps(rubric))
