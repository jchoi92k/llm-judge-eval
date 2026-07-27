# tests/test_persistence.py

"""
Tests for JSON checkpoint persistence of evaluations
(Evaluator._save_evaluations / _load_existing_evaluations and the
serialization helpers in utils) — including automatic migration of
legacy pickle checkpoints. Synthetic data only.
"""

import json
import logging
import pickle
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation_pipeline import utils
from evaluation_pipeline.evaluator import Evaluator

PARSED_EVAL = {
    "scores": {"category_a": {"sub_1": 3}},
    "mathematical_accuracy_relevance": {"applicable": True},
}


class FakeSDKResponse:
    """Pydantic-like SDK response: has model_dump() and usage attributes.

    Module-level so instances are picklable (legacy checkpoint fixtures).
    """

    def __init__(self, text="{}", input_tokens=100, output_tokens=50, cached_tokens=20):
        self.output_text = text
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        )

    def model_dump(self):
        return {
            "output_text": self.output_text,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "input_tokens_details": {
                    "cached_tokens": self.usage.input_tokens_details.cached_tokens
                },
            },
        }


class PersistenceStub:
    """Duck-typed Evaluator with just the state the persistence methods touch."""

    _save_evaluations = Evaluator._save_evaluations
    _load_existing_evaluations = Evaluator._load_existing_evaluations

    def __init__(self, tmp_path, run_id="testrun"):
        self.config = SimpleNamespace(
            run_id=run_id,
            dirs=SimpleNamespace(evaluation_results=tmp_path),
        )
        self.logger = logging.getLogger(run_id)
        self.evaluations = defaultdict(list)


def json_checkpoint_path(tmp_path, run_id="testrun"):
    return tmp_path / f"{run_id}_evaluations.json"


# ============================================================================
# SAVE / LOAD ROUND-TRIP
# ============================================================================

class TestSaveLoadRoundTrip:

    def test_save_writes_json_checkpoint(self, tmp_path):
        ev = PersistenceStub(tmp_path)
        ev.evaluations["s1"].append([PARSED_EVAL, FakeSDKResponse()])
        ev._save_evaluations()

        path = json_checkpoint_path(tmp_path)
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["format"] == utils.EVALUATIONS_FORMAT
        assert not path.with_name(path.name + ".tmp").exists()

    def test_round_trip_preserves_parsed_evals_and_usage(self, tmp_path):
        ev = PersistenceStub(tmp_path)
        ev.evaluations["s1"].append([PARSED_EVAL, FakeSDKResponse(input_tokens=7, output_tokens=3, cached_tokens=1)])
        ev.evaluations["s1"].append([PARSED_EVAL, FakeSDKResponse()])
        ev._save_evaluations()

        loaded = PersistenceStub(tmp_path)
        loaded._load_existing_evaluations()
        assert set(loaded.evaluations) == {"s1"}
        assert len(loaded.evaluations["s1"]) == 2
        assert loaded.evaluations["s1"][0][0] == PARSED_EVAL
        assert utils.response_usage(loaded.evaluations["s1"][0][1]) == (7, 3, 1)

    def test_loaded_evaluations_behave_as_defaultdict(self, tmp_path):
        ev = PersistenceStub(tmp_path)
        ev.evaluations["s1"].append([PARSED_EVAL, FakeSDKResponse()])
        ev._save_evaluations()

        loaded = PersistenceStub(tmp_path)
        loaded._load_existing_evaluations()
        # Resume logic indexes unseen sessions directly; must yield [] not KeyError
        assert loaded.evaluations["never_seen"] == []

    def test_no_evaluations_saves_nothing(self, tmp_path):
        ev = PersistenceStub(tmp_path)
        ev._save_evaluations()
        assert not json_checkpoint_path(tmp_path).exists()

    def test_load_with_no_checkpoint_leaves_evaluations_empty(self, tmp_path):
        ev = PersistenceStub(tmp_path)
        ev._load_existing_evaluations()
        assert len(ev.evaluations) == 0


# ============================================================================
# SESSION ID TYPES
# ============================================================================

class TestSessionIdTypes:

    def test_int_session_ids_round_trip_as_ints(self, tmp_path):
        ev = PersistenceStub(tmp_path)
        ev.evaluations[42].append([PARSED_EVAL, FakeSDKResponse()])
        ev._save_evaluations()

        loaded = PersistenceStub(tmp_path)
        loaded._load_existing_evaluations()
        assert 42 in loaded.evaluations
        assert "42" not in loaded.evaluations

    def test_numpy_int_session_ids_saved_as_plain_ints(self, tmp_path):
        ev = PersistenceStub(tmp_path)
        ev.evaluations[np.int64(7)].append([PARSED_EVAL, FakeSDKResponse()])
        ev._save_evaluations()

        loaded = PersistenceStub(tmp_path)
        loaded._load_existing_evaluations()
        # np.int64(7) and 7 hash equally, so resume lookups keep working
        assert len(loaded.evaluations[np.int64(7)]) == 1


# ============================================================================
# LEGACY PICKLE MIGRATION
# ============================================================================

class TestPickleMigration:

    def write_legacy_pickle(self, tmp_path, evaluations, run_id="testrun"):
        pkl_path = tmp_path / f"{run_id}_evaluations.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(evaluations, f)
        return pkl_path

    def test_pkl_migrates_to_json_and_keeps_backup(self, tmp_path):
        legacy = defaultdict(list)
        legacy["s1"].append([PARSED_EVAL, FakeSDKResponse(input_tokens=11, output_tokens=5, cached_tokens=2)])
        pkl_path = self.write_legacy_pickle(tmp_path, legacy)
        pkl_bytes = pkl_path.read_bytes()

        ev = PersistenceStub(tmp_path)
        ev._load_existing_evaluations()

        assert json_checkpoint_path(tmp_path).exists()
        assert pkl_path.read_bytes() == pkl_bytes  # original untouched
        assert ev.evaluations["s1"][0][0] == PARSED_EVAL
        # In-memory responses become plain dicts, same as a JSON load
        assert isinstance(ev.evaluations["s1"][0][1], dict)
        assert utils.response_usage(ev.evaluations["s1"][0][1]) == (11, 5, 2)

    def test_json_preferred_over_pkl_when_both_exist(self, tmp_path):
        stale = defaultdict(list)
        stale["stale_session"].append([PARSED_EVAL, FakeSDKResponse()])
        self.write_legacy_pickle(tmp_path, stale)

        fresh = PersistenceStub(tmp_path)
        fresh.evaluations["fresh_session"].append([PARSED_EVAL, FakeSDKResponse()])
        fresh._save_evaluations()

        loaded = PersistenceStub(tmp_path)
        loaded._load_existing_evaluations()
        assert set(loaded.evaluations) == {"fresh_session"}


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

class TestSerializeResponse:

    def test_dict_passes_through(self):
        d = {"usage": {"input_tokens": 1}}
        assert utils.serialize_response(d) == d

    def test_model_dump_used_when_available(self):
        r = FakeSDKResponse(text="hello", input_tokens=9)
        out = utils.serialize_response(r)
        assert out["output_text"] == "hello"
        assert out["usage"]["input_tokens"] == 9

    def test_dict_with_non_json_values_is_coerced(self):
        d = {"created_at": np.int64(123), "path": SimpleNamespace(x=1)}
        out = utils.serialize_response(d)
        json.dumps(out)  # must be JSON-safe

    def test_object_without_model_dump_falls_back_to_vars(self):
        obj = SimpleNamespace(output_text="abc", usage=None)
        out = utils.serialize_response(obj)
        assert out["output_text"] == "abc"


class TestPayloadFormat:

    def test_unknown_format_rejected(self):
        with pytest.raises(ValueError):
            utils.payload_to_evaluations({"format": "something-else", "sessions": []})


class TestResponseUsage:

    def test_sdk_object_usage(self):
        r = FakeSDKResponse(input_tokens=10, output_tokens=4, cached_tokens=3)
        assert utils.response_usage(r) == (10, 4, 3)

    def test_dict_usage(self):
        d = {"usage": {"input_tokens": 8, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 5}}}
        assert utils.response_usage(d) == (8, 2, 5)

    def test_missing_usage_counts_as_zero(self):
        assert utils.response_usage({}) == (0, 0, 0)
        assert utils.response_usage(SimpleNamespace(usage=None)) == (0, 0, 0)

    def test_dict_usage_without_details(self):
        d = {"usage": {"input_tokens": 8, "output_tokens": 2}}
        assert utils.response_usage(d) == (8, 2, 0)


# ============================================================================
# COST REPORTING OVER JSON CHECKPOINTS
# ============================================================================

class TestReportActualCostWithJson:

    def make_reporting_stub(self, tmp_path):
        ev = PersistenceStub(tmp_path)
        ev.config.model = SimpleNamespace(
            input_token_price=1.0, cached_token_price=0.5, output_token_price=2.0
        )
        return ev

    def test_reads_json_checkpoint_and_counts_dict_usage(self, tmp_path, capsys):
        from evaluation_pipeline import reporting

        ev = self.make_reporting_stub(tmp_path)
        ev.evaluations["s1"].append([PARSED_EVAL, FakeSDKResponse(input_tokens=10, output_tokens=4, cached_tokens=2)])
        ev._save_evaluations()

        # Fresh stub so the data comes from the JSON file, not memory
        fresh = self.make_reporting_stub(tmp_path)
        results = reporting.report_actual_cost(fresh)
        assert len(results) == 1
        assert results[0]["input_tokens"] == 10
        assert results[0]["output_tokens"] == 4
        assert results[0]["cached_tokens"] == 2

    def test_all_runs_dedupes_json_and_pkl_for_same_run(self, tmp_path):
        from evaluation_pipeline import reporting

        ev = self.make_reporting_stub(tmp_path)
        ev.evaluations["s1"].append([PARSED_EVAL, FakeSDKResponse()])
        ev._save_evaluations()
        # Leave a legacy pickle for the same run_id alongside the JSON
        with open(tmp_path / "testrun_evaluations.pkl", "wb") as f:
            pickle.dump(dict(ev.evaluations), f)

        fresh = self.make_reporting_stub(tmp_path)
        results = reporting.report_actual_cost(fresh, all_runs=True)
        assert [r["run_id"] for r in results] == ["testrun"]
