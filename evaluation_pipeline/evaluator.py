# evaluation_pipeline/evaluator.py

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from .config import Config
from .openai_client import OpenAIClient
from .prompts import PromptBuilder, retrieve_similar
from . import calibration, data, execution, guidelines, reporting, scoring, utils


class Evaluator:
    """
    Main evaluation pipeline orchestrator.

    Coordinates guideline generation, prompt building, API calls,
    adjudication, and final score generation.

    Feature logic lives in plain-function modules (guidelines, execution,
    scoring, calibration, reporting); the methods here are thin delegates
    so the public API stays unchanged.

    Args:
        config: Configuration object; see Config class and CONFIG.md for details.
        data_prep_function: Optional custom data preprocessing function; applies to session data after loading.
    """

    def __init__(
            self,
            config: Config,
            data_prep_function: Callable[[pd.DataFrame, Config], pd.DataFrame] | None = None
            ):

        # Initialize logger and components
        self.config = config
        self.logger = logging.getLogger(config.run_id)
        self.data_prep_function = data_prep_function

        # Initialize client and prompt builder
        self.client = OpenAIClient(config)
        self.prompt_builder = PromptBuilder(config)

        # Data containers
        self.session_data: Optional[pd.DataFrame] = None
        self.human_evaluation: Optional[pd.DataFrame] = None
        self.human_evaluation_curated: Optional[pd.DataFrame] = None
        self.rag_dictionary: Optional[Dict] = None
        self.rag_embeddings: Optional[Dict] = None

        # Evaluation state
        self.evaluations: Dict[str, List] = defaultdict(list)
        self.guidelines: Dict[str, str] = {}
        self.dynamic_prompts: Dict[str, List] = {}

        # Failures skipped during evaluation runs (parse errors, batch issues).
        # Diagnostic, in-memory only; appended across runs in this session.
        self.failures: List[Dict] = []

        # Batch processing components
        self.batch_file_path: Optional[Path] = None
        self.batch_id: Optional[str] = None

        # Load existing data if available
        self._load_existing_evaluations()
        self._load_existing_guidelines()
        self._auto_load_data()

        # For development/testing
        self.test_components = []

    def __repr__(self):
        """
        Display checklist-style status of the evaluation pipeline.
        ✓ = Completed; good to go
        △ = In progress or optional
        ✗ = Not started or missing; action needed
        """
        return reporting.status_text(self)

    def _aggregate_texts(self, path: Path) -> str:
        """
        Aggregate all text files from all subdirectories with subdirectory names prepended.
        Currently only used to aggregate all DWW practice guides.
        Add relevant context files and subdirectories to the 'practice_guides' path defined in config.toml if needed
        """
        return guidelines.aggregate_texts(path)

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def load_session_data(self):
        """Load and preprocess session data."""
        if self.session_data is not None:
            self.logger.info("Session data already loaded, skipping")
            return self
        self.logger.info("Loading session data...")
        self.session_data = data.load_session_data(self.config, self.data_prep_function)
        self.logger.info(f"Loaded {len(self.session_data)} sessions")
        return self

    def load_human_evaluation(self):
        """Load human evaluation data; optional."""
        if self.human_evaluation is not None:
            self.logger.info("Human evaluation already loaded, skipping")
            return self
        self.logger.info("Loading human evaluation data...")
        self.human_evaluation = data.load_human_evaluation(self.config)
        self.logger.info(f"Loaded {len(self.human_evaluation)} human evaluations")
        return self

    def load_rag_data(self):
        """Load RAG dictionary and embeddings."""
        if self.rag_dictionary is not None and self.rag_embeddings is not None:
            self.logger.info("RAG data already loaded, skipping")
            return self
        self.logger.info("Loading RAG data...")
        self.rag_dictionary, self.rag_embeddings = data.load_rag_data(self.config)
        self.logger.info(f"Loaded {len(self.rag_dictionary)} RAG entries")
        return self

    def _load_existing_evaluations(self):
        """Load existing evaluations from disk if available."""
        eval_file_path = self.config.dirs.evaluation_results / f"{self.config.run_id}_evaluations.pkl"
        if eval_file_path.exists():
            self.logger.info(f"Loading existing evaluations from {eval_file_path}")
            with open(eval_file_path, "rb") as f:
                self.evaluations = pickle.load(f)
            self.logger.info(f"Loaded evaluations for {len(self.evaluations)} sessions")
        return self

    def _load_existing_guidelines(self):
        """Load existing guidelines from disk if available."""
        guideline_pattern = f"guideline_*.txt"
        guideline_files = list(self.config.dirs.evaluation_guidelines.glob(guideline_pattern))

        for file_path in guideline_files:
            self.guidelines[file_path.name] = file_path.read_text(encoding='utf-8')

        if self.guidelines:
            self.logger.info(f"Loaded {len(self.guidelines)} existing guideline(s)")

        return self

    def _auto_load_data(self):
        """Auto-load data on initialization. Raises error if critical data missing."""
        # Load session data (REQUIRED)
        try:
            self.session_data = data.load_session_data(self.config, self.data_prep_function)
            if self.session_data is None or len(self.session_data) == 0:
                raise ValueError("Session data is empty or None")
            self.logger.info(f"Auto-loaded {len(self.session_data)} sessions")
        except FileNotFoundError as e:
            raise ValueError(f"Session data file not found: {e}")
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"Session data file is empty: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading session data: {e}")
            raise ValueError(f"Failed to load session data (REQUIRED): {e}")

        # Load RAG data (REQUIRED)
        try:
            self.rag_dictionary, self.rag_embeddings = data.load_rag_data(self.config)
            if not self.rag_dictionary or not self.rag_embeddings:
                raise ValueError("RAG data is empty or None")
            self.logger.info(f"Auto-loaded RAG data ({len(self.rag_dictionary)} entries)")
        except FileNotFoundError as e:
            raise ValueError(f"RAG data file not found: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading RAG data: {e}")
            raise ValueError(f"Failed to load RAG data (REQUIRED): {e}")

        # Load human evaluation (OPTIONAL)
        try:
            self.human_evaluation = data.load_human_evaluation(self.config)
            self.logger.info(f"Auto-loaded human evaluation ({len(self.human_evaluation)} samples)")
        except Exception as e:
            self.logger.warning(f"Human evaluation not loaded: {e}")
            self.logger.warning(f"You may proceed without human evaluations, but it may affect guideline and evaluation quality.")
            self.human_evaluation = None

        # Load curated few-shot examples (OPTIONAL — falls back to human_evaluation)
        curated_path = getattr(self.config.file_paths, 'human_evaluation_curated', None)
        if curated_path and curated_path.exists():
            self.human_evaluation_curated = pd.read_csv(curated_path)
            if 'image_data_base64' in self.human_evaluation_curated.columns:
                self.human_evaluation_curated = self.human_evaluation_curated.drop(columns=['image_data_base64'])
            self.logger.info(f"Auto-loaded curated few-shot examples ({len(self.human_evaluation_curated)} samples)")
        else:
            self.human_evaluation_curated = None

        return self

    def _save_evaluations(self):
        """Save evaluations to disk. Path defined by config."""
        if not self.evaluations:
            self.logger.warning("No evaluations to save")
            return

        eval_file_path = self.config.dirs.evaluation_results / f"{self.config.run_id}_evaluations.pkl"

        # Ensure directory exists
        eval_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(eval_file_path, "wb") as f:
                pickle.dump(self.evaluations, f)
            self.logger.info(f"Saved evaluations to {eval_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save evaluations: {e}")
            raise

    # ========================================================================
    # GUIDELINE GENERATION
    # ========================================================================

    def generate_evaluation_guidelines(self, service_tier: str = "flex", n_runs: int = 2, auto_approve: bool = False, force_regenerate: bool = False, test_run: bool = False):
        """
        Generate evaluation guidelines using multiple LLM runs + aggregation.

        Args:
            n_runs: Number of guideline variations to generate (currently fixed at 2)
            auto_approve: If True, skip cost confirmation prompt
            force_regenerate: If True, regenerate even if guidelines exist

        Returns:
            Self for potential method chaining
        """
        return guidelines.generate_evaluation_guidelines(self, service_tier=service_tier, n_runs=n_runs, auto_approve=auto_approve, force_regenerate=force_regenerate, test_run=test_run)

    # ========================================================================
    # DYNAMIC PROMPT GENERATION
    # ========================================================================

    def generate_dynamic_prompts(self, adjudication: bool = False):
        """
        Generate evaluation prompts for each session.

        Args:
            adjudication: If True, generate adjudication prompts for sessions
                         that need it. If False, generate evaluation prompts.

        Returns:
            Self for method chaining
        """
        if self.session_data is None:
            raise ValueError("Session data not loaded. Call load_session_data() first.")

        if self.rag_dictionary is None or self.rag_embeddings is None:
            raise ValueError("RAG data not loaded. Call load_rag_data() first.")

        # Check if guidelines exist
        if not self.guidelines.get("guideline_final.txt"):
            raise ValueError("Guidelines not generated. Call generate_evaluation_guidelines() first.")

        # Reset dynamic prompts
        self.dynamic_prompts = {}

        # Get final guideline
        final_guideline = self.guidelines.get("guideline_final.txt", "")

        # Determine which sessions to process
        if adjudication:
            # Find sessions that need adjudication
            target_ids = []
            for session_id in self.evaluations:
                # Need at least 2 evaluations to check for adjudication
                if len(self.evaluations[session_id]) < 2:
                    continue

                # Check all pairs of evaluations for discrepancies
                needs_adj = False
                for i in range(len(self.evaluations[session_id]) - 1):
                    for j in range(i + 1, len(self.evaluations[session_id])):
                        adj_needed, reason = utils.needs_adjudication(
                            self.evaluations[session_id][i][0],
                            self.evaluations[session_id][j][0]
                        )
                        if adj_needed:
                            needs_adj = True
                            self.logger.debug(f"Session {session_id} needs adjudication: {reason}")
                            break
                    if needs_adj:
                        break

                if needs_adj:
                    target_ids.append(session_id)

            target_data = self.session_data[self.session_data['session_id'].isin(target_ids)]

            if len(target_data) == 0:
                self.logger.info("No sessions require adjudication")
                return self
        else:
            # Process sessions that don't have 2+ evaluations yet
            evaluated_sessions = [
                sid for sid, evals in self.evaluations.items()
                if len(evals) >= 2
            ]
            target_data = self.session_data[
                ~self.session_data['session_id'].isin(evaluated_sessions)
            ]

        if len(target_data) == 0:
            status = self.check_evaluation_status()
            if adjudication:
                self.logger.info("No sessions require adjudication. All evaluations agree or are incomplete.")
            else:
                self.logger.info(f"No sessions require prompt generation. Status: {status['next_action']}")
            return self

        self.logger.info(f"Generating {'adjudication' if adjudication else 'evaluation'} prompts for {len(target_data)} sessions...")

        # Generate prompts for each session
        for _, row in tqdm(target_data.iterrows(), total=len(target_data), desc="Generating prompts"):
            session_id = row['session_id']

            # Format row for embedding; exclude the image data row (urls, base64, etc.)
            row_string = utils.format_any_tabular_data(
                row[~row.index.isin(['image_data_base64'])],
                f"{self.config.tool_settings.tool_name} Data"
            )

            # for text-embedding-3-small, limit to 8000 tokens
            row_string = utils.truncate_text_to_tokens(row_string, max_tokens=8000)

            # Get embedding and retrieve similar RAG content
            embedding = self.client.create_embedding(row_string)
            similar_entries = retrieve_similar(embedding, self.rag_embeddings, top_k=1)
            similar_cid = similar_entries[0][0]
            rag_context = self.rag_dictionary.get(similar_cid, "No similar content found.")

            # Build appropriate prompt
            if adjudication:
                prompt = self.prompt_builder.build_adjudication_prompt(
                    row=row,
                    rag_context=rag_context,
                    guideline=final_guideline,
                    evaluation_1=self.evaluations[session_id][0][0],
                    evaluation_2=self.evaluations[session_id][1][0]
                )
            else:
                prompt = self.prompt_builder.build_evaluation_prompt(
                    row=row,
                    rag_context=rag_context,
                    guideline=final_guideline,
                    human_evaluation_samples=self.human_evaluation_curated if self.human_evaluation_curated is not None else self.human_evaluation,
                )

            self.dynamic_prompts[session_id] = prompt

        self.logger.info(f"Generated {len(self.dynamic_prompts)} prompts")
        return self

    def sanity_check_prompts(self, n_examples: int = 3):
        """
        Sanity check generated prompts for correct few-shot insertion and image handling.

        Checks:
        1. Few-shot examples: Whether the "Example Evaluations from Human Raters" section
           contains actual content (not empty).
        2. Images: Whether sessions with image data have input_image content parts in the prompt.

        Args:
            n_examples: Number of example sessions to display for each check.
        """
        return reporting.sanity_check_prompts(self, n_examples=n_examples)

    # ========================================================================
    # FLEX EVALUATION (Direct API calls)
    # ========================================================================

    def flex_evaluate(self, service_tier: str = "flex", adjudication: bool = False, n_runs: int = 2, auto_approve: bool = False):
        """
        Run evaluations using direct API calls with flex pricing.

        Args:
            n_runs: Number of evaluation runs per session; expected to be 2 for normal evaluation; overridden to 1 for adjudication
            adjudication: Whether the run is an adjudication run
            service_tier: Service tier to use for API calls; can be 'flex', 'auto', or 'default'
            auto_approve: If True, skip cost confirmation

        Returns:
            Self for method chaining
        """
        return execution.flex_evaluate(self, service_tier=service_tier, adjudication=adjudication, n_runs=n_runs, auto_approve=auto_approve)

    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================

    def prepare_batch_file(self, adjudication: bool = False, n_runs: int = 2, auto_approve: bool = False):
        """
        Prepare JSONL batch file for OpenAI batch API.

        Args:
            adjudication: Whether the run is an adjudication run
            n_runs: Number of evaluation runs per session; expected to be 2 for normal evaluation; overridden to 1 for adjudication
            auto_approve: If True, skip cost confirmation prompt

        Returns:
            Self for method chaining
        """
        return execution.prepare_batch_file(self, adjudication=adjudication, n_runs=n_runs, auto_approve=auto_approve)

    def upload_batch(self):
        """
        Upload batch file to OpenAI and create batch job.

        Returns:
            Self for method chaining
        """
        return execution.upload_batch(self)

    def check_batch_status(self) -> str:
        """
        Check the status of the current batch job.

        Returns:
            Status string
        """
        return execution.check_batch_status(self)

    def check_and_retrieve(self, until_complete: bool = False, check_interval: int = 60, batch_id_override: Optional[str] = None):
        """
        Check batch status and optionally wait until complete + retrieve results.

        Args:
            until_complete: If True, wait and check periodically until done.
                        If False, check once and return.
            check_interval: Seconds between status checks (only used if until_complete=True)
            batch_id_override: Optional batch ID to use instead of self.batch_id

        Returns:
            Self for method chaining
        """
        return execution.check_and_retrieve(self, until_complete=until_complete, check_interval=check_interval, batch_id_override=batch_id_override)

    def _retrieve_batch_results(self, batch_id_override: Optional[str] = None):
        """Retrieve and process batch results."""
        return execution.retrieve_batch_results(self, batch_id_override=batch_id_override)

    def cancel_batch(self):
        """Cancel the current batch job."""
        return execution.cancel_batch(self)

    # ========================================================================
    # FINAL SCORE GENERATION
    # ========================================================================

    def generate_final_scores(self):
        """
        Generate final scores using adjudication if available, otherwise average.

        Returns:
            Self for method chaining
        """
        return scoring.generate_final_scores(self)

    # ========================================================================
    # STATUS AND DIAGNOSTICS
    # ========================================================================

    def check_evaluation_status(self) -> Dict[str, Any]:
        """
        Check the status of evaluations across all sessions.

        Returns:
            Dictionary with evaluation status information
        """
        return scoring.check_evaluation_status(self)

    def failures_summary(self) -> list:
        """
        Print a summary of evaluations that were skipped during this session's
        runs (parse failures, batch retrieval issues) and return the raw
        failure entries. In-memory only; cleared when the Evaluator is recreated.
        """
        return reporting.failures_summary(self)

    def report_actual_cost(self, all_runs: bool = False) -> Dict[str, Any]:
        """
        Report actual token usage and cost from completed evaluations.

        Extracts usage data from OpenAI response objects stored in evaluations.
        Uses model pricing from config for cost calculation.

        Args:
            all_runs: If True, scan ALL pkl files in evaluation_results dir.
                      If False, only report on current run's evaluations.

        Returns:
            Dictionary with usage and cost breakdown.
        """
        return reporting.report_actual_cost(self, all_runs=all_runs)

    # ========================================================================
    # SNIFF TEST / ERROR ANALYSIS
    # ========================================================================

    def _generate_sniff_test_prompts(self, sniff_data: pd.DataFrame):
        """
        Generate evaluation prompts for sniff test sessions with per-session
        few-shot exclusion: each session's prompt includes human-reviewed
        examples from OTHER sessions, never its own scores.

        The few-shot pool is self.human_evaluation. For each session being
        evaluated, we exclude that session from the pool and sample from the
        remainder (up to n_human_rating_samples from config).
        """
        return calibration.generate_sniff_test_prompts(self, sniff_data)

    def run_sniff_test(
        self,
        service_tier: str = "flex",
        auto_approve: bool = False,
        human_id_col: str = "session_id",
        column_map: dict = None,
    ) -> dict:
        """
        Run evaluation on the human-reviewed subset and compare with human scores.

        This is a lightweight calibration step between guideline generation and
        full evaluation. It evaluates only the sessions that have human review
        data, then generates an error analysis report with guideline amendment
        recommendations.

        Few-shot handling: Each session's evaluation prompt includes human-reviewed
        examples from OTHER sessions only, never its own scores.

        Args:
            service_tier: API service tier ("flex", "auto", "default")
            auto_approve: Skip cost confirmation prompts
            human_id_col: Column name for session IDs in human evaluation data
            column_map: Custom mapping from human CSV column names to rubric format.
                       Defaults to error_analysis.DEFAULT_HUMAN_COLUMN_MAP.

        Returns:
            Dict with keys: comparison_df, metrics, llm_analysis, report
        """
        return calibration.run_sniff_test(self, service_tier=service_tier, auto_approve=auto_approve, human_id_col=human_id_col, column_map=column_map)

    def refine_guidelines(
        self,
        sniff_test_report: str,
        service_tier: str = "flex",
        guideline_name: str = "guideline_refined.txt",
    ) -> str:
        """
        Use the sniff test error analysis report to refine the evaluation guidelines.

        Sends the current guidelines + the error analysis report to the LLM,
        asking it to produce an amended version of the guidelines. The refined
        guidelines are saved alongside the original (not overwriting it).

        Args:
            sniff_test_report: The full error analysis report (from run_sniff_test)
            service_tier: API service tier
            guideline_name: Filename for the refined guidelines

        Returns:
            The refined guideline text
        """
        return calibration.refine_guidelines(self, sniff_test_report, service_tier=service_tier, guideline_name=guideline_name)

    def use_refined_guidelines(self, guideline_name: str = "guideline_refined.txt"):
        """
        Swap the active final guideline to the refined version.

        This copies the refined guideline over guideline_final.txt so subsequent
        evaluation steps use the refined version.

        Args:
            guideline_name: Name of the refined guideline file to activate
        """
        return calibration.use_refined_guidelines(self, guideline_name=guideline_name)

    # ========================================================================
    # END-TO-END PIPELINE
    # ========================================================================

    def run(self,
            mode: str = "flex",
            auto_approve: bool = False,
            n_runs: int = 2,
            skip_adjudication: bool = False,
            check_interval: int = 60,
            force_regenerate_guidelines: bool = False):
        """
        Run the complete evaluation pipeline end-to-end.

        Args:
            mode: "flex" for direct API calls or "batch" for batch processing
            auto_approve: Skip all cost confirmations
            n_runs: Number of evaluation runs per session
            skip_adjudication: Skip adjudication step
            check_interval: For batch mode, seconds between status checks
            force_regenerate_guidelines: Force guideline regeneration even if they exist
        """
        self.logger.info("Starting evaluation pipeline...")

        status = self.check_evaluation_status()

        # Check: skip straight to final scores
        if status['ready_for_final_scores'] and not status['needs_adjudication'] and not status['not_started']:
            self.logger.info("All evaluations complete. Generating final scores...")
            self.generate_final_scores()
            return self

        # Even in batch mode, generate evaluation guidelines using flex/default/auto
        guideline_service_tier = "flex"

        # Guidelines
        self.generate_evaluation_guidelines(
            auto_approve=auto_approve,
            force_regenerate=force_regenerate_guidelines,
            service_tier=guideline_service_tier
        )

        # Evaluation
        self.generate_dynamic_prompts()

        if mode in ("flex", "auto", "default"):
            self.flex_evaluate(adjudication=False, service_tier=mode, n_runs=n_runs, auto_approve=auto_approve)
        elif mode == "batch":
            self.prepare_batch_file(adjudication=False, n_runs=n_runs, auto_approve=auto_approve)
            self.upload_batch()
            self.check_and_retrieve(until_complete=True, check_interval=check_interval)

        # Adjudication
        if not skip_adjudication:
            # Check if any sessions need adjudication
            status = self.check_evaluation_status()

            if status['needs_adjudication']:
                self.logger.info(f"{len(status['needs_adjudication'])} session(s) need adjudication")
                self.generate_dynamic_prompts(adjudication=True)

                if mode in ("flex", "auto", "default"):
                    self.flex_evaluate(adjudication=True, service_tier=mode, auto_approve=auto_approve)
                elif mode == "batch":
                    self.prepare_batch_file(adjudication=True, auto_approve=auto_approve)
                    self.upload_batch()
                    self.check_and_retrieve(until_complete=True, check_interval=check_interval)
            else:
                self.logger.info("No sessions require adjudication")

        # Final scores
        self.generate_final_scores()

        self.logger.info("Pipeline complete!")
        return self
