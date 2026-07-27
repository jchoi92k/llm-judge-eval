# tests/test_execution_loop.py

"""
Integration tests for the evaluation execution loop (evaluation_pipeline.execution)
using a fake OpenAI client — no network calls, no real data.

These tests pin the loop's observable behavior: run counts, resume/skip logic,
parse-failure handling, per-session checkpointing, and batch file preparation
and retrieval (including the custom_id → session mapping).
"""

import json
import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest

from evaluation_pipeline import execution

VALID_EVAL = {
    "scores": {"category_a": {"sub_1": 3}},
    "mathematical_accuracy_relevance": {"applicable": True},
}


def make_prompt(text):
    return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]


class FakeResponse:
    def __init__(self, text):
        self.output_text = text
        self.usage = None


class FakeClient:
    """Stands in for OpenAIClient: canned responses, zero cost, no network."""

    def __init__(self, responses=None, batch_results=None):
        self.calls = []
        self.call_kwargs = []
        self._responses = list(responses) if responses else None
        self._batch_results = batch_results or []

    def call(self, prompt, service_tier="flex", **kwargs):
        self.calls.append(prompt)
        self.call_kwargs.append(kwargs)
        if self._responses is not None:
            return self._responses.pop(0)
        return FakeResponse(json.dumps(VALID_EVAL))

    def estimate_cost(self, prompt_cached, prompt_uncached, expected_output_tokens=None):
        return 0.0

    def retrieve_batch_results(self, batch_id):
        return self._batch_results


class StubEvaluator:
    """Duck-typed Evaluator with just the state the execution functions touch."""

    def __init__(self, tmp_path, client=None):
        self.config = SimpleNamespace(
            run_id="testrun",
            model=SimpleNamespace(model_name="gpt-test", input_token_price=0.0),
            api_settings=SimpleNamespace(use_structured_outputs=True),
            dirs=SimpleNamespace(
                batch_processing=tmp_path,
                batch_processing_results=tmp_path,
            ),
        )
        self.logger = logging.getLogger("testrun")
        self.client = client or FakeClient()
        self.prompt_builder = SimpleNamespace(
            output_json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        )
        self.evaluations = defaultdict(list)
        self.dynamic_prompts = {}
        self.failures = []
        self.batch_file_path = None
        self.batch_id = None
        self.save_count = 0

    def _save_evaluations(self):
        self.save_count += 1


def batch_result(custom_id, eval_dict=VALID_EVAL):
    """Build a batch API result line in the shape retrieve_batch_results expects."""
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(eval_dict)}
                        ],
                    }
                ]
            }
        },
    }


# ============================================================================
# FLEX EVALUATION LOOP
# ============================================================================

def test_flex_normal_mode_runs_twice_per_session(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}

    execution.flex_evaluate(ev, auto_approve=True)

    assert len(ev.evaluations["s1"]) == 2
    assert len(ev.evaluations["s2"]) == 2
    assert len(ev.client.calls) == 4
    assert ev.evaluations["s1"][0][0] == VALID_EVAL


def test_flex_saves_after_each_session(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}

    execution.flex_evaluate(ev, auto_approve=True)

    assert ev.save_count == 2


def test_flex_resume_skips_fully_evaluated_sessions(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}
    ev.evaluations["s1"] = [[VALID_EVAL, None], [VALID_EVAL, None]]

    execution.flex_evaluate(ev, auto_approve=True)

    assert len(ev.evaluations["s1"]) == 2  # untouched
    assert len(ev.evaluations["s2"]) == 2
    assert len(ev.client.calls) == 2  # only s2's two runs


def test_flex_resume_completes_partially_evaluated_session(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}
    ev.evaluations["s1"] = [[VALID_EVAL, None]]

    execution.flex_evaluate(ev, auto_approve=True)

    assert len(ev.evaluations["s1"]) == 2
    assert len(ev.client.calls) == 1  # only the missing second run


def test_flex_adjudication_mode_runs_once_per_session(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}
    ev.evaluations["s1"] = [[VALID_EVAL, None], [VALID_EVAL, None]]
    ev.evaluations["s2"] = [[VALID_EVAL, None], [VALID_EVAL, None]]

    execution.flex_evaluate(ev, adjudication=True, auto_approve=True)

    assert len(ev.evaluations["s1"]) == 3
    assert len(ev.evaluations["s2"]) == 3
    assert len(ev.client.calls) == 2


def test_flex_parse_failure_is_skipped_without_crashing(tmp_path):
    client = FakeClient(responses=[
        FakeResponse("this is not json at all {{{"),
        FakeResponse(json.dumps(VALID_EVAL)),
    ])
    ev = StubEvaluator(tmp_path, client=client)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}

    execution.flex_evaluate(ev, auto_approve=True)

    # First run failed to parse and was dropped; second succeeded
    assert len(ev.evaluations["s1"]) == 1
    assert len(ev.client.calls) == 2


def test_flex_raises_without_dynamic_prompts(tmp_path):
    ev = StubEvaluator(tmp_path)

    with pytest.raises(ValueError, match="No dynamic prompts"):
        execution.flex_evaluate(ev, auto_approve=True)


# ============================================================================
# FAILURES TRACKING
# ============================================================================

def test_flex_parse_failure_recorded_in_failures(tmp_path):
    client = FakeClient(responses=[
        FakeResponse("not json {{{"),
        FakeResponse(json.dumps(VALID_EVAL)),
    ])
    ev = StubEvaluator(tmp_path, client=client)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}

    execution.flex_evaluate(ev, auto_approve=True)

    assert len(ev.failures) == 1
    failure = ev.failures[0]
    assert failure["session_id"] == "s1"
    assert failure["reason"] == "parse_failure"
    assert "evaluation run" in failure["stage"]


def test_flex_success_records_no_failures(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}

    execution.flex_evaluate(ev, auto_approve=True)

    assert ev.failures == []


def test_adjudication_parse_failure_recorded(tmp_path):
    client = FakeClient(responses=[FakeResponse("garbage")])
    ev = StubEvaluator(tmp_path, client=client)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}

    execution.flex_evaluate(ev, adjudication=True, auto_approve=True)

    assert len(ev.failures) == 1
    assert ev.failures[0]["stage"] == "adjudication"


def test_batch_parse_failure_recorded_with_session(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}
    execution.prepare_batch_file(ev, auto_approve=True)

    entries = [json.loads(l) for l in ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")]
    results = [batch_result(entries[0]["custom_id"])]
    bad = batch_result(entries[1]["custom_id"])
    bad["response"]["body"]["output"][0]["content"][0]["text"] = "garbage {{{"
    results.append(bad)
    ev.client._batch_results = results
    ev.batch_id = "batch_123"

    execution.retrieve_batch_results(ev)

    assert len(ev.failures) == 1
    assert ev.failures[0] == {
        "session_id": "s1",
        "stage": "batch",
        "reason": "parse_failure",
        "detail": ev.failures[0]["detail"],  # error message text not pinned
    }


def test_batch_unmapped_custom_id_recorded(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}
    execution.prepare_batch_file(ev, auto_approve=True)

    ev.client._batch_results = [batch_result("someone_elses_custom_id_99")]
    ev.batch_id = "batch_123"

    execution.retrieve_batch_results(ev)

    assert len(ev.failures) == 1
    assert ev.failures[0]["reason"] == "unmapped_custom_id"
    assert ev.failures[0]["session_id"] is None


def test_failures_summary_prints_and_returns(tmp_path, capsys):
    from evaluation_pipeline import reporting

    ev = StubEvaluator(tmp_path)
    ev.failures = [
        {"session_id": "s1", "stage": "evaluation run 1/2", "reason": "parse_failure", "detail": "bad json"},
        {"session_id": None, "stage": "batch", "reason": "missing_custom_id", "detail": "no id"},
    ]

    returned = reporting.failures_summary(ev)

    out = capsys.readouterr().out
    assert "2 total" in out
    assert "parse_failure: 1" in out
    assert "missing_custom_id: 1" in out
    assert "s1" in out
    assert returned == ev.failures


def test_failures_summary_empty(tmp_path, capsys):
    from evaluation_pipeline import reporting

    ev = StubEvaluator(tmp_path)
    returned = reporting.failures_summary(ev)

    assert "No recorded failures" in capsys.readouterr().out
    assert returned == []


# ============================================================================
# STRUCTURED OUTPUTS
# ============================================================================

def test_flex_passes_output_schema_to_client(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}

    execution.flex_evaluate(ev, auto_approve=True)

    assert all(
        kw.get("output_schema") == ev.prompt_builder.output_json_schema
        for kw in ev.client.call_kwargs
    )


def test_flex_omits_schema_when_structured_outputs_disabled(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.config.api_settings.use_structured_outputs = False
    ev.dynamic_prompts = {"s1": make_prompt("p1")}

    execution.flex_evaluate(ev, auto_approve=True)

    assert all(kw.get("output_schema") is None for kw in ev.client.call_kwargs)


def test_batch_bodies_include_text_format_schema(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}

    execution.prepare_batch_file(ev, auto_approve=True)

    entries = [json.loads(l) for l in ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")]
    for e in entries:
        fmt = e["body"]["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["strict"] is True
        assert fmt["schema"] == ev.prompt_builder.output_json_schema


def test_batch_bodies_omit_text_format_when_disabled(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.config.api_settings.use_structured_outputs = False
    ev.dynamic_prompts = {"s1": make_prompt("p1")}

    execution.prepare_batch_file(ev, auto_approve=True)

    entries = [json.loads(l) for l in ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")]
    assert all("text" not in e["body"] for e in entries)


# ============================================================================
# BATCH FILE PREPARATION
# ============================================================================

def test_prepare_batch_file_writes_n_runs_entries_per_session(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}

    execution.prepare_batch_file(ev, auto_approve=True)

    lines = ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4
    entries = [json.loads(l) for l in lines]
    assert all(e["url"] == "/v1/responses" for e in entries)
    assert all(e["body"]["model"] == "gpt-test" for e in entries)
    assert len({e["custom_id"] for e in entries}) == 4  # unique custom_ids


def test_prepare_batch_file_adjudication_writes_one_entry_per_session(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}

    execution.prepare_batch_file(ev, adjudication=True, auto_approve=True)

    lines = ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


# ============================================================================
# BATCH RESULT RETRIEVAL
# ============================================================================

def test_retrieve_batch_results_normal_two_run_batch(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}
    execution.prepare_batch_file(ev, auto_approve=True)

    entries = [json.loads(l) for l in ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")]
    ev.client._batch_results = [batch_result(e["custom_id"]) for e in entries]
    ev.batch_id = "batch_123"

    execution.retrieve_batch_results(ev)

    assert len(ev.evaluations["s1"]) == 2
    assert len(ev.evaluations["s2"]) == 2
    assert ev.save_count == 1


def test_prepare_batch_file_writes_custom_id_mapping(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}

    execution.prepare_batch_file(ev, auto_approve=True)

    mapping_path = execution._mapping_path_for(ev.batch_file_path)
    assert mapping_path.exists()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    entries = [json.loads(l) for l in ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")]
    assert set(mapping.keys()) == {e["custom_id"] for e in entries}
    assert sorted(mapping.values()) == ["s1", "s1", "s2", "s2"]


def test_retrieve_adjudication_batch_maps_to_correct_sessions(tmp_path):
    """Regression test: adjudication batches have 1 request/session, but legacy
    retrieval assumed 2, silently mapping results to the wrong sessions."""
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {
        "s1": make_prompt("p1"),
        "s2": make_prompt("p2"),
        "s3": make_prompt("p3"),
        "s4": make_prompt("p4"),
    }
    execution.prepare_batch_file(ev, adjudication=True, auto_approve=True)

    mapping = json.loads(execution._mapping_path_for(ev.batch_file_path).read_text(encoding="utf-8"))
    # Distinct payload per session so misrouting would be detected
    ev.client._batch_results = [
        batch_result(cid, {"scores": {"cat": {"sub": 1}}, "session_marker": sid})
        for cid, sid in mapping.items()
    ]
    ev.batch_id = "batch_adj"

    execution.retrieve_batch_results(ev)

    for sid in ["s1", "s2", "s3", "s4"]:
        assert len(ev.evaluations[sid]) == 1
        assert ev.evaluations[sid][0][0]["session_marker"] == sid


def test_retrieve_legacy_batch_without_mapping_falls_back(tmp_path):
    """Batches created before the mapping sidecar existed still retrieve via
    request-order arithmetic (2 runs/session)."""
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1"), "s2": make_prompt("p2")}
    execution.prepare_batch_file(ev, auto_approve=True)

    # Simulate a legacy batch: delete the sidecar
    execution._mapping_path_for(ev.batch_file_path).unlink()

    entries = [json.loads(l) for l in ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")]
    ev.client._batch_results = [batch_result(e["custom_id"]) for e in entries]
    ev.batch_id = "batch_legacy"

    execution.retrieve_batch_results(ev)

    assert len(ev.evaluations["s1"]) == 2
    assert len(ev.evaluations["s2"]) == 2


def test_retrieve_batch_results_parse_failure_counted_not_crashing(tmp_path):
    ev = StubEvaluator(tmp_path)
    ev.dynamic_prompts = {"s1": make_prompt("p1")}
    execution.prepare_batch_file(ev, auto_approve=True)

    entries = [json.loads(l) for l in ev.batch_file_path.read_text(encoding="utf-8").strip().split("\n")]
    results = [batch_result(entries[0]["custom_id"])]
    bad = batch_result(entries[1]["custom_id"])
    bad["response"]["body"]["output"][0]["content"][0]["text"] = "garbage {{{"
    results.append(bad)
    ev.client._batch_results = results
    ev.batch_id = "batch_123"

    execution.retrieve_batch_results(ev)

    assert len(ev.evaluations["s1"]) == 1
