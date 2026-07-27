# tests/test_config.py

"""
Tests for config additions from task 6 (magic numbers → config), focused on
run_id stability: existing partner configs must keep their run_id, while a
changed sampling seed must produce a different hash input.
"""

from pathlib import Path

from evaluation_pipeline.config import (
    APISettings,
    Config,
    Directories,
    EvaluationSettings,
    FilePaths,
    ModelConfig,
    ToolSettings,
)


def make_config(**eval_overrides) -> Config:
    """Build a Config without touching the filesystem (validators bypassed)."""
    eval_kwargs = {"n_samples": 5, "n_human_rating_samples": 3}
    eval_kwargs.update(eval_overrides)
    return Config.model_construct(
        evaluation_settings=EvaluationSettings(**eval_kwargs),
        model=ModelConfig(
            model_name="test-model",
            price_per_1M_input_tokens=1.0,
            price_per_1M_cached_input_tokens=0.5,
            price_per_1M_output_tokens=2.0,
        ),
        tool_settings=ToolSettings(tool_name="testtool"),
        api_settings=APISettings(),
        file_paths=FilePaths.model_construct(session_data=Path("x.csv")),
        dirs=Directories.model_construct(),
    )


class TestRunIdStability:
    def test_default_random_state_omitted_from_hash_input(self):
        """Pre-existing configs (no random_state key in toml) must produce the
        same to_dict as before the field existed — else every partner run_id
        changes and their checkpoints are orphaned."""
        d = make_config().to_dict()
        assert "random_state" not in d["evaluation_settings"]

    def test_custom_random_state_included_in_hash_input(self):
        """A non-default seed changes what data gets sampled, so it must
        change the run_id hash input."""
        d = make_config(random_state=7).to_dict()
        assert d["evaluation_settings"]["random_state"] == 7

    def test_default_and_custom_seed_hash_inputs_differ(self):
        assert make_config().to_dict() != make_config(random_state=7).to_dict()

    def test_api_settings_still_excluded_from_hash_input(self):
        """Cost-estimation constants and structured-outputs toggle live in
        api_settings precisely because it is not hashed."""
        assert "api_settings" not in make_config().to_dict()


class TestNewDefaults:
    def test_cost_estimation_defaults_match_old_hardcoded_values(self):
        api = APISettings()
        assert api.expected_evaluation_output_tokens == 500
        assert api.tokens_per_image == 1100

    def test_random_state_default_matches_old_hardcoded_value(self):
        assert EvaluationSettings(n_samples=1, n_human_rating_samples=1).random_state == 42
