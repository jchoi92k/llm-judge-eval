# tests/test_scoring.py

"""
Tests for evaluation_pipeline.scoring.generate_final_scores — pins the core
scoring contract, especially the "third evaluation is the adjudication and
wins outright" assumption that other components are built around.
"""

import json
import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest

from evaluation_pipeline import scoring


def make_eval(validity=3, feedback=3):
    return {
        "scores": {
            "Mathematical_Accuracy": {"Validity": validity},
            "Pedagogical_Quality": {"Feedback": feedback},
        },
        "explanations": {
            "Mathematical_Accuracy": {"Validity": "reason"},
            "Pedagogical_Quality": {"Feedback": "reason"},
        },
        "mathematical_accuracy_relevance": {"applicable": True},
    }


class StubEvaluator:
    def __init__(self, tmp_path):
        self.config = SimpleNamespace(
            run_id="testrun",
            dirs=SimpleNamespace(evaluation_results=tmp_path),
        )
        self.logger = logging.getLogger("testrun")
        self.evaluations = defaultdict(list)


def read_final_scores(tmp_path):
    return json.loads((tmp_path / "testrun_final_scores.json").read_text(encoding="utf-8"))


def test_third_eval_is_adjudication_and_wins_outright(tmp_path):
    """Core assumption: with 3 evaluations, the third (adjudication) is taken
    verbatim — the first two are ignored entirely, no averaging."""
    ev = StubEvaluator(tmp_path)
    ev.evaluations["s1"] = [
        [make_eval(validity=1, feedback=1), None],
        [make_eval(validity=4, feedback=4), None],
        [make_eval(validity=2, feedback=3), None],  # adjudication
    ]

    scoring.generate_final_scores(ev)

    final = read_final_scores(tmp_path)
    assert final["s1"]["scores"]["Mathematical_Accuracy"]["Validity"] == 2
    assert final["s1"]["scores"]["Pedagogical_Quality"]["Feedback"] == 3


def test_two_evals_are_averaged_and_rounded(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.evaluations["s1"] = [
        [make_eval(validity=1, feedback=2), None],
        [make_eval(validity=2, feedback=3), None],
    ]

    scoring.generate_final_scores(ev)

    final = read_final_scores(tmp_path)
    # round((1+2)/2) = round(1.5) = 2 (banker's rounding: 1.5 -> 2)
    assert final["s1"]["scores"]["Mathematical_Accuracy"]["Validity"] == round(1.5)
    assert final["s1"]["scores"]["Pedagogical_Quality"]["Feedback"] == round(2.5)


def test_null_scores_average_to_the_non_null_value(tmp_path):
    ev = StubEvaluator(tmp_path)
    e1, e2 = make_eval(), make_eval()
    e1["scores"]["Mathematical_Accuracy"]["Validity"] = None
    e2["scores"]["Mathematical_Accuracy"]["Validity"] = 4

    ev.evaluations["s1"] = [[e1, None], [e2, None]]
    scoring.generate_final_scores(ev)

    assert read_final_scores(tmp_path)["s1"]["scores"]["Mathematical_Accuracy"]["Validity"] == 4


def test_both_null_stays_null(tmp_path):
    ev = StubEvaluator(tmp_path)
    e1, e2 = make_eval(), make_eval()
    e1["scores"]["Mathematical_Accuracy"]["Validity"] = None
    e2["scores"]["Mathematical_Accuracy"]["Validity"] = None

    ev.evaluations["s1"] = [[e1, None], [e2, None]]
    scoring.generate_final_scores(ev)

    assert read_final_scores(tmp_path)["s1"]["scores"]["Mathematical_Accuracy"]["Validity"] is None


def test_string_scores_coerced_before_averaging(tmp_path):
    """Legacy pkl data can hold scores as strings."""
    ev = StubEvaluator(tmp_path)
    e1, e2 = make_eval(), make_eval()
    e1["scores"]["Mathematical_Accuracy"]["Validity"] = "2"
    e2["scores"]["Mathematical_Accuracy"]["Validity"] = "4"

    ev.evaluations["s1"] = [[e1, None], [e2, None]]
    scoring.generate_final_scores(ev)

    assert read_final_scores(tmp_path)["s1"]["scores"]["Mathematical_Accuracy"]["Validity"] == 3


def test_single_eval_session_excluded_and_reported(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.evaluations["s1"] = [[make_eval(), None]]
    ev.evaluations["s2"] = [[make_eval(validity=2), None], [make_eval(validity=2), None]]

    scoring.generate_final_scores(ev)

    final = read_final_scores(tmp_path)
    assert "s1" not in final
    assert "s2" in final
    incomplete = json.loads((tmp_path / "testrun_incomplete_sessions.json").read_text(encoding="utf-8"))
    assert [e["session_id"] for e in incomplete] == ["s1"]


def test_mismatched_structures_excluded_as_incomplete(tmp_path):
    ev = StubEvaluator(tmp_path)
    e2 = make_eval()
    del e2["scores"]["Pedagogical_Quality"]
    ev.evaluations["s1"] = [[make_eval(), None], [e2, None]]

    scoring.generate_final_scores(ev)

    assert read_final_scores(tmp_path) == {}
    incomplete = json.loads((tmp_path / "testrun_incomplete_sessions.json").read_text(encoding="utf-8"))
    assert incomplete[0]["issue"] == "Mismatched evaluation structure"


def test_no_evaluations_raises(tmp_path):
    ev = StubEvaluator(tmp_path)
    with pytest.raises(ValueError, match="No evaluations found"):
        scoring.generate_final_scores(ev)


def test_non_score_fields_come_from_first_eval_when_averaging(tmp_path):
    ev = StubEvaluator(tmp_path)
    e1, e2 = make_eval(), make_eval()
    e1["explanations"]["Mathematical_Accuracy"]["Validity"] = "from eval 1"
    e2["explanations"]["Mathematical_Accuracy"]["Validity"] = "from eval 2"

    ev.evaluations["s1"] = [[e1, None], [e2, None]]
    scoring.generate_final_scores(ev)

    final = read_final_scores(tmp_path)
    assert final["s1"]["explanations"]["Mathematical_Accuracy"]["Validity"] == "from eval 1"
