# evaluation_pipeline/evaluator.py

import logging
import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import sys

import pandas as pd
from tqdm import tqdm

from .config import Config
from .openai_client import OpenAIClient
from .prompts import PromptBuilder, retrieve_similar
from . import data, utils


class Evaluator:
    """
    Main evaluation pipeline orchestrator.
    
    Coordinates guideline generation, prompt building, API calls,
    adjudication, and final score generation.
    
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
        lines = [f"Evaluator(run_id={self.config.run_id}, model={self.config.model.model_name})"]
        lines.append("\nStatus:")
        
        # Check data loading
        if self.session_data is not None:
            lines.append(f"  ✓ Session data loaded ({len(self.session_data)} sessions)")
        else:
            lines.append("  ✗ Session data not loaded")
        
        # Check if human evaluation exists; optional
        if self.human_evaluation is not None:
            lines.append(f"  ✓ Human evaluation loaded ({len(self.human_evaluation)} samples) [optional]")
        else:
            lines.append("  △ Human evaluation not loaded [optional]")
        
        # Check RAG data load
        if self.rag_dictionary is not None and self.rag_embeddings is not None:
            lines.append(f"  ✓ RAG data loaded ({len(self.rag_dictionary)} entries)")
        else:
            lines.append("  ✗ RAG data not loaded")
        
        # Check if guidelines have already been generated
        if "guideline_final.txt" in self.guidelines:
            lines.append("  ✓ Guidelines available (shared across runs)")
        else:
            lines.append("  ✗ Guidelines not generated")
        
        # Check evaluations using check_evaluation_status
        if self.session_data is not None:
            status = self.check_evaluation_status()
            
            # Display general evaluation progress
            total = status['total_sessions']
            complete = len(status['complete'])
            needs_adj = len(status['needs_adjudication'])
            ready = len(status['ready_for_final_scores'])
            
            if complete == total:
                lines.append(f"  ✓ Evaluations complete ({complete}/{total} sessions)")
            elif complete > 0:
                lines.append(f"  △ Evaluations in progress ({complete}/{total} sessions complete)")
            else:
                lines.append(f"  ✗ Evaluations not started (0/{total} sessions)")
            
            if needs_adj > 0:
                lines.append(f"    → {needs_adj} session(s) need adjudication")
            if ready > 0:
                lines.append(f"    → {ready} session(s) ready for final scores")
        
        # Next steps
        lines.append("\nNext steps:")

        if self.session_data is not None:
            status = self.check_evaluation_status()
            
            # Check if guidelines exist
            guidelines_exist = "guideline_final.txt" in self.guidelines
            
            # If guidelines exist, show evaluation next steps
            if guidelines_exist:
                lines.append(f"  → {status['next_action']}")

            # If no guidelines, check human eval status
            elif self.human_evaluation is None:
                lines.append("  → Provide human evaluation data via config.toml (optional but recommended)")
                lines.append("  → Or proceed to run generate_evaluation_guidelines()")
            else:
                lines.append("  → Run generate_evaluation_guidelines()")
        
        return "\n".join(lines)
    
    def _aggregate_texts(self, path: Path) -> str:
        """
        Aggregate all text files from all subdirectories with subdirectory names prepended.
        Currently only used to aggregate all DWW practice guides.
        Add relevant context files and subdirectories to the 'practice_guides' path defined in config.toml if needed
        """
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            results = []
            for subdir in subdirs:
                texts = [f.read_text(encoding='utf-8', errors='ignore') for f in subdir.glob('*.txt')]
                if texts:
                    results.append(f"{subdir.name}\n" + '\n'.join(texts))
                else:
                    results.append(f"{subdir.name}\n(no text files)")
            return '\n\n'.join(results)
        
        raise ValueError(f"No subdirectories found in practice guides path: {path}")

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
    
    def build_guidelines_aggregation_prompt(
        self,
        original_prompt: str,
        guideline_outputs: List[str],
    ) -> str:
        """Build prompt for aggregating multiple guideline outputs."""
        return self.guidelines_aggregation_template.render(
            ORIGINAL_PROMPT=original_prompt,
            outputs=guideline_outputs,
            enumerate=enumerate
        )

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
        # Define guideline file names (shared across runs)
        # Guideline names are hardcoded
        guideline_file_names = [f"guideline_run_{i+1}.txt" for i in range(n_runs)]
        guideline_file_names.append("guideline_final.txt")
        
        # Check if final guideline already exists
        final_guideline_path = self.config.dirs.evaluation_guidelines / "guideline_final.txt"
        
        if final_guideline_path.exists() and not force_regenerate:
            self.logger.info("Existing guidelines found. Loading from file.")
            self.logger.info("Guidelines are not tied to specific runs. To regenerate, use force_regenerate=True or delete guideline files.")
            
            # Load existing guidelines
            for file_name in guideline_file_names:
                file_path = self.config.dirs.evaluation_guidelines / file_name
                if file_path.exists():
                    self.guidelines[file_name] = file_path.read_text(encoding='utf-8')
            
            return self
        
        # If force_regenerate, clear existing guidelines
        if force_regenerate:
            try:
                self.logger.info("Force regeneration requested. Deleting existing guidelines.")
                for file_name in guideline_file_names:
                    file_path = self.config.dirs.evaluation_guidelines / file_name
                    if file_path.exists():
                        file_path.unlink()
                self.guidelines.clear()
            # If error due to file open etc., log and raise
            except Exception as e:
                self.logger.error(f"Failed to delete existing guidelines: {e}")
                raise
        
        # Load existing guidelines (for partial regeneration)
        for file_name in guideline_file_names:
            file_path = self.config.dirs.evaluation_guidelines / file_name
            if file_path.exists():
                self.guidelines[file_name] = file_path.read_text(encoding='utf-8')
        
        # If all exist, skip generation
        if len(self.guidelines) == len(guideline_file_names):
            self.logger.info("All guidelines already exist, skipping generation")
            return self
        
        # Aggregate practice guides
        if test_run:
            practice_guides = "Test practice guide content."
        else:
            practice_guides = self._aggregate_texts(self.config.dirs.practice_guides)
        
        # Build guideline prompt
        if self.human_evaluation is None:
            if not auto_approve:
                utils.flush_logs()
                user_input = input("Human evaluation data not loaded. Proceed without human evaluation? (y/n): ")
                if user_input.lower() != 'y':
                    self.logger.info("Guideline generation cancelled")
                    return self

            # use n=1 samples of session data to show sample data sans human evaluations
            sample_data = self.session_data.sample(n=1, random_state=42).reset_index(drop=True)
            
            # remove image data from sample data for now
            if 'image_data_base64' in sample_data.columns:
                sample_data = sample_data.drop(columns=['image_data_base64'])

            guideline_prompt = self.prompt_builder.build_guideline_prompt(sample_data, practice_guides)
        else:
            guideline_prompt = self.prompt_builder.build_guideline_prompt(self.human_evaluation, practice_guides)
        
        # TEST LINE
        self.test_components.append(guideline_prompt)

        # Rough cost
        guideline_run_1 = self.client.estimate_cost(
            prompt_cached="", 
            prompt_uncached=guideline_prompt,
            expected_output_tokens=self.config.model.expected_output_tokens
        ) 
        
        guideline_run_2 = self.client.estimate_cost(
            prompt_cached=guideline_prompt, 
            prompt_uncached="",
            expected_output_tokens=self.config.model.expected_output_tokens
        ) 
        
        guideline_run_adjudcation = self.client.estimate_cost(
            prompt_cached="", 
            prompt_uncached=guideline_prompt,
            expected_output_tokens=self.config.model.expected_output_tokens
        ) 
        
        guideline_cost = guideline_run_1 + guideline_run_2 + guideline_run_adjudcation
        
        self.logger.info(f"Estimated guideline generation cost: ${guideline_cost:.4f}")
        utils.flush_logs()
        
        # Get approval
        if not auto_approve:
            user_input = input("Proceed with guideline generation? (y/n): ")
            if user_input.lower() != 'y':
                self.logger.info("Guideline generation cancelled")
                return self
        
        # Generate guidelines
        for i in range(n_runs):
            file_name = f"guideline_run_{i+1}.txt"
            if file_name in self.guidelines:
                continue
            
            self.logger.info(f"Generating guideline {i+1}/{n_runs}...")
            response = self.client.call(guideline_prompt, service_tier=service_tier)
            guideline_text = response.output_text.strip()
            
            self.guidelines[file_name] = guideline_text
            
            # Save to disk
            file_path = self.config.dirs.evaluation_guidelines / file_name
            file_path.write_text(guideline_text, encoding='utf-8')
        
        # Generate final aggregated guideline
        final_file_name = "guideline_final.txt"
        if final_file_name not in self.guidelines:
            self.logger.info("Generating final aggregated guideline...")
            
            guideline_outputs = [v for k, v in self.guidelines.items() if "guideline_final" not in k]
            
            adjudication_prompt = self.prompt_builder.build_guidelines_aggregation_prompt(
                original_prompt=guideline_prompt,
                guideline_outputs=guideline_outputs,
            )
            
            response = self.client.call(adjudication_prompt, service_tier=service_tier)
            final_guideline = response.output_text.strip()
            
            self.guidelines[final_file_name] = final_guideline
            file_path = self.config.dirs.evaluation_guidelines / final_file_name
            file_path.write_text(final_guideline, encoding='utf-8')
        
            # TEST LINE
            self.test_components.append(adjudication_prompt)

        self.logger.info("Guideline generation complete")
        return self
    
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
        if not self.dynamic_prompts:
            print("No dynamic prompts generated. Run generate_dynamic_prompts() first.")
            return

        print(f"{'='*70}")
        print(f"PROMPT SANITY CHECK — {len(self.dynamic_prompts)} prompts")
        print(f"{'='*70}\n")

        # ---- Check 1: Few-shot examples ----
        print("## 1. Few-Shot Human Examples\n")

        few_shot_marker = "### Example Evaluations from Human Raters"
        next_section_marker = "### Example Tutor Conversations"

        has_examples = 0
        empty_examples = 0
        example_char_counts = []

        for session_id, prompt in self.dynamic_prompts.items():
            # Extract the full text from the prompt structure
            full_text = ""
            for msg in prompt:
                if isinstance(msg, dict) and "content" in msg:
                    for part in msg["content"]:
                        if part.get("type") == "input_text":
                            full_text += part["text"] + "\n"

            # Find the few-shot section
            start = full_text.find(few_shot_marker)
            if start == -1:
                empty_examples += 1
                example_char_counts.append((session_id, 0))
                continue

            end = full_text.find(next_section_marker, start)
            section_text = full_text[start + len(few_shot_marker):end if end != -1 else None].strip()
            char_count = len(section_text)
            example_char_counts.append((session_id, char_count))

            if char_count > 50:  # non-trivial content
                has_examples += 1
            else:
                empty_examples += 1

        print(f"  Sessions with few-shot content:  {has_examples}/{len(self.dynamic_prompts)}")
        print(f"  Sessions without few-shot content: {empty_examples}/{len(self.dynamic_prompts)}")

        if has_examples == 0:
            print(f"\n  ⚠ WARNING: No prompts contain few-shot examples!")
        elif empty_examples > 0:
            print(f"\n  ⚠ WARNING: {empty_examples} prompts are missing few-shot examples")

        # Show examples
        sorted_counts = sorted(example_char_counts, key=lambda x: x[1], reverse=True)
        print(f"\n  Top {min(n_examples, len(sorted_counts))} by few-shot section size:")
        for sid, count in sorted_counts[:n_examples]:
            print(f"    {str(sid)[:20]}...  {count:,} chars")

        if empty_examples > 0:
            print(f"\n  Empty few-shot examples:")
            empty = [x for x in sorted_counts if x[1] <= 50]
            for sid, count in empty[:n_examples]:
                print(f"    {str(sid)[:20]}...  {count} chars")

        # ---- Check 2: Image handling ----
        print(f"\n## 2. Image Handling\n")

        sessions_with_images = 0
        prompts_with_images = 0
        image_mismatches = []
        image_examples = []

        for _, row in self.session_data.iterrows():
            session_id = row["session_id"]
            if session_id not in self.dynamic_prompts:
                continue

            img_data = row.get("image_data_base64", None)
            has_image_data = (
                img_data is not None
                and not isinstance(img_data, float)
                and isinstance(img_data, list)
                and any(x is not None for x in img_data)
            )
            n_images_data = len([x for x in img_data if x is not None]) if has_image_data else 0

            if has_image_data:
                sessions_with_images += 1

            # Count image parts in prompt
            prompt = self.dynamic_prompts[session_id]
            n_images_prompt = 0
            for msg in prompt:
                if isinstance(msg, dict) and "content" in msg:
                    for part in msg["content"]:
                        if part.get("type") == "input_image":
                            n_images_prompt += 1

            if n_images_prompt > 0:
                prompts_with_images += 1

            if has_image_data and n_images_prompt == 0:
                image_mismatches.append((session_id, n_images_data, n_images_prompt, "data has images, prompt does not"))
            elif not has_image_data and n_images_prompt > 0:
                image_mismatches.append((session_id, n_images_data, n_images_prompt, "prompt has images, data does not"))
            elif has_image_data and n_images_data != n_images_prompt:
                image_mismatches.append((session_id, n_images_data, n_images_prompt, "count mismatch"))

            if has_image_data or n_images_prompt > 0:
                image_examples.append((session_id, n_images_data, n_images_prompt))

        print(f"  Sessions with image data:    {sessions_with_images}/{len(self.dynamic_prompts)}")
        print(f"  Prompts with image parts:    {prompts_with_images}/{len(self.dynamic_prompts)}")

        if image_mismatches:
            print(f"\n  ⚠ MISMATCHES FOUND: {len(image_mismatches)}")
            for sid, n_data, n_prompt, reason in image_mismatches[:n_examples]:
                print(f"    {str(sid)[:20]}...  data: {n_data} images, prompt: {n_prompt} images — {reason}")
        else:
            if sessions_with_images > 0:
                print(f"\n  ✓ All image counts match between data and prompts")
            else:
                print(f"\n  (No sessions with images in current prompt set)")

        if image_examples:
            print(f"\n  Sample sessions with images (up to {n_examples}):")
            for sid, n_data, n_prompt in image_examples[:n_examples]:
                prompt = self.dynamic_prompts[sid]
                # Show content part types
                part_types = []
                for msg in prompt:
                    if isinstance(msg, dict) and "content" in msg:
                        for part in msg["content"]:
                            ptype = part.get("type", "?")
                            if ptype == "input_image":
                                url = part.get("image_url", "")
                                part_types.append(f"input_image({url[:40]}...)")
                            elif ptype == "input_text":
                                part_types.append(f"input_text({len(part.get('text',''))} chars)")
                print(f"\n    {sid[:30]}...  (data: {n_data}, prompt: {n_prompt})")
                for pt in part_types:
                    print(f"      {pt}")

        print(f"\n{'='*70}")

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
        if not self.dynamic_prompts:
            raise ValueError("No dynamic prompts generated. Call generate_dynamic_prompts() first. Or you might be in adjudication mode with no sessions needing adjudication.")
        
        # Find common cached prefix
        all_prompts = list(self.dynamic_prompts.values())
        if all_prompts:

            cached_prefix, uncached_text = utils.find_prefix(all_prompts)

            text_cost_per_eval = self.client.estimate_cost(
                prompt_cached=cached_prefix,
                prompt_uncached=uncached_text,
                expected_output_tokens=500  # evaluation output estimate hardcoded for now
            )

            # Estimate image token cost
            image_counts = [utils.count_images_in_prompt(p) for p in all_prompts]
            avg_images = sum(image_counts) / len(image_counts) if image_counts else 0
            avg_image_tokens = utils.estimate_image_tokens(int(avg_images))
            image_cost_per_eval = avg_image_tokens * self.config.model.input_token_price

            cost_per_evaluation = text_cost_per_eval + image_cost_per_eval
            total_cost = cost_per_evaluation * len(self.dynamic_prompts) * n_runs

            n_with_images = sum(1 for c in image_counts if c > 0)
            self.logger.info(
                f"Estimated evaluation cost: ${total_cost:.4f} "
                f"(text: ${text_cost_per_eval * len(self.dynamic_prompts) * n_runs:.4f}, "
                f"images: ${image_cost_per_eval * len(self.dynamic_prompts) * n_runs:.4f} — "
                f"{n_with_images}/{len(all_prompts)} prompts with images, avg {avg_images:.1f} images/prompt)"
            )

            if not auto_approve:
                utils.flush_logs()
                user_input = input("Proceed with evaluation? (y/n): ")
                if user_input.lower() != 'y':
                    self.logger.info("Evaluation cancelled")
                    return self

        if adjudication:
            n_runs = 1
            self.logger.info("Adjudication mode: using 1 run per session")
        else:
            self.logger.info(f"Evaluation mode: using {n_runs} runs per session")

        # Run evaluations
        for session_id, prompt in tqdm(self.dynamic_prompts.items(), desc="Evaluating sessions"):
            if adjudication:
                # Adjudication mode: always run exactly 1 evaluation
                
                self.logger.debug(f"Running adjudication for {session_id}")
                
                response = self.client.call(prompt, service_tier=service_tier)
                eval_text = response.output_text.strip()
                
                success, parsed_eval, error_msg = utils.try_parse_evaluation(eval_text)
                
                if not success:
                    self.logger.error(f"Failed to parse adjudication for {session_id}: {error_msg}")
                    continue
                
                self.evaluations[session_id].append([parsed_eval, response])
                
            else:
                # Regular evaluation mode: run n_runs evaluations
                # Skip if already have enough evaluations
                if len(self.evaluations[session_id]) >= n_runs:
                    continue
                
                for run in range(n_runs):
                    # Skip if we already have this run
                    if len(self.evaluations[session_id]) > run:
                        continue
                    
                    self.logger.debug(f"Evaluating {session_id}, run {run + 1}/{n_runs}")
                    
                    response = self.client.call(prompt, service_tier=service_tier)
                    eval_text = response.output_text.strip()
                    
                    success, parsed_eval, error_msg = utils.try_parse_evaluation(eval_text)
                    
                    if not success:
                        self.logger.error(f"Failed to parse evaluation for {session_id}, run {run + 1}: {error_msg}")
                        continue
                    
                    self.evaluations[session_id].append([parsed_eval, response])
            
            # Save after each session
            self._save_evaluations()
        
        self.logger.info("Flex evaluation complete")
        return self
    
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
        if not self.dynamic_prompts:
            raise ValueError("No dynamic prompts generated. Call generate_dynamic_prompts() first.")
        
        timestamp = int(time.time())
        self.batch_file_path = self.config.dirs.batch_processing / f"{self.config.run_id}_{timestamp}_batch.jsonl"
        
        if self.batch_file_path.exists():
            self.logger.info("Batch file already exists, skipping creation")
            return self

        # Find common cached prefix
        all_prompts = list(self.dynamic_prompts.values())
        if all_prompts:
            cached_prefix, uncached_text = utils.find_prefix(all_prompts)

            text_cost_per_eval = self.client.estimate_cost(
                prompt_cached=cached_prefix,
                prompt_uncached=uncached_text,
                expected_output_tokens=500  # evaluation output estimate hardcoded for now
            )

            # Estimate image token cost
            image_counts = [utils.count_images_in_prompt(p) for p in all_prompts]
            avg_images = sum(image_counts) / len(image_counts) if image_counts else 0
            avg_image_tokens = utils.estimate_image_tokens(int(avg_images))
            image_cost_per_eval = avg_image_tokens * self.config.model.input_token_price

            cost_per_evaluation = text_cost_per_eval + image_cost_per_eval
            total_cost = cost_per_evaluation * len(self.dynamic_prompts) * n_runs

            n_with_images = sum(1 for c in image_counts if c > 0)
            self.logger.info(
                f"Estimated batch processing cost: ${total_cost:.4f} "
                f"(text: ${text_cost_per_eval * len(self.dynamic_prompts) * n_runs:.4f}, "
                f"images: ${image_cost_per_eval * len(self.dynamic_prompts) * n_runs:.4f} — "
                f"{n_with_images}/{len(all_prompts)} prompts with images, avg {avg_images:.1f} images/prompt)"
            )

            if not auto_approve:
                utils.flush_logs()
                user_input = input("Proceed with batch file creation? (y/n): ")
                if user_input.lower() != 'y':
                    self.logger.info("Batch file creation cancelled")
                    return self

        request_counter = 0

        if adjudication:
            n_runs = 1
            self.logger.info("Adjudication mode: using 1 run per session")
        else:
            self.logger.info(f"Evaluation mode: using {n_runs} runs per session")

        for _, prompt in self.dynamic_prompts.items():
            for _ in range(n_runs):
                batch_entry = {
                    "custom_id": f"{self.config.run_id}_{timestamp}_{request_counter}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.config.model.model_name,
                        "input": prompt
                    }
                }
                
                with open(self.batch_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(batch_entry) + "\n")
                
                request_counter += 1
        
        self.logger.info(f"Created batch file at {self.batch_file_path} with {request_counter} requests")
        return self
    
    def upload_batch(self):
        """
        Upload batch file to OpenAI and create batch job.
        
        Returns:
            Self for method chaining
        """
        if not self.batch_file_path or not self.batch_file_path.exists():
            raise ValueError("Batch file not found. Call prepare_batch_file() first.")
        
        self.batch_id = self.client.upload_batch_file(self.batch_file_path)
        self.logger.info(f"Batch uploaded with ID: {self.batch_id}")
        return self
    
    def check_batch_status(self) -> str:
        """
        Check the status of the current batch job.

        Returns:
            Status string
        """
        if not self.batch_id:
            raise ValueError("No batch ID found. Call upload_batch() first.")

        info = self.client.check_batch_status(self.batch_id)
        status = info["status"]
        self.logger.info(f"Batch status: {status} ({info['completed']}/{info['total']} completed, {info['failed']} failed)")
        return status
    
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
        # Use override if provided, otherwise use self.batch_id
        batch_id = batch_id_override or self.batch_id
        
        if not batch_id:
            raise ValueError("No batch ID found. Call upload_batch() first or provide batch_id_override.")
        
        if not until_complete:
            info = self.client.check_batch_status(batch_id)
            self.logger.info(f"Batch status: {info['status']} ({info['completed']}/{info['total']} completed, {info['failed']} failed)")
            return self

        # Wait until complete
        while True:
            info = self.client.check_batch_status(batch_id)
            status = info["status"]
            self.logger.info(f"Batch status: {status} ({info['completed']}/{info['total']} completed, {info['failed']} failed)")

            if status == "completed":
                self.logger.info("Batch completed successfully")
                self._retrieve_batch_results(batch_id)
                break
            elif status in ["failed", "cancelled"]:
                self.logger.error(f"Batch ended with status: {status}")
                break
            else:
                self.logger.info(f"Checking again in {check_interval} seconds...")
                time.sleep(check_interval)
        
        return self

    def _retrieve_batch_results(self, batch_id_override: Optional[str] = None):
        """Retrieve and process batch results."""
        batch_id = batch_id_override or self.batch_id
        
        if not batch_id:
            raise ValueError("No batch ID found")
    
        self.logger.info("Retrieving batch results...")
        results = self.client.retrieve_batch_results(batch_id)

        # Save raw results
        output_filename = self.config.dirs.batch_processing_results / f"batch_results_{self.config.run_id}.pkl"
        with open(output_filename, 'wb') as f:
            pickle.dump({'results': results}, f)
        
        # Process results into evaluations
        session_ids = list(self.dynamic_prompts.keys())
        n_runs = 2 # Assuming 2 runs per session for evaluation
        processed_count = 0
        failed_count = 0
        
        for result in results:
            try:
                custom_id = result.get('custom_id', '')
                if not custom_id:
                    self.logger.warning("Missing custom_id in batch result")
                    failed_count += 1
                    continue
                    
                counter = int(custom_id.split('_')[-1])
                session_idx = counter // n_runs
                
                if session_idx >= len(session_ids):
                    self.logger.warning(f"Session index {session_idx} out of range (max: {len(session_ids)-1})")
                    failed_count += 1
                    continue
                
                session_id = session_ids[session_idx]
                
                # Safely extract evaluation text from response
                eval_text = None
                response_body = result.get('response', {}).get('body', {})
                
                if response_body.get('output'):
                    output = response_body['output']
                    
                    for item in output:
                        if item.get('type') == 'message' and item.get('content'):
                            for content_item in item['content']:
                                if content_item.get('type') == 'output_text':
                                    eval_text = content_item.get('text', '').strip()
                                    break
                            break
                
                if eval_text:
                    success, parsed_eval, error_msg = utils.try_parse_evaluation(eval_text)
                    
                    if success:
                        self.evaluations[session_id].append([parsed_eval, result['response']])
                        processed_count += 1
                    else:
                        self.logger.error(f"Failed to parse evaluation for {session_id}: {error_msg}")
                        failed_count += 1
                else:
                    self.logger.warning(f"No evaluation text found for session {session_id}")
                    failed_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing batch result: {e}")
                failed_count += 1
                continue
        
        # Save processed evaluations
        self._save_evaluations()
        self.logger.info(f"Batch results processed and saved: {processed_count} successful, {failed_count} failed")
    
    def cancel_batch(self):
        """Cancel the current batch job."""
        if not self.batch_id:
            raise ValueError("No batch ID found")
        
        self.client.cancel_batch(self.batch_id)
        self.logger.info(f"Batch {self.batch_id} cancelled")
        return self
    
    # ========================================================================
    # FINAL SCORE GENERATION
    # ========================================================================
    
    def generate_final_scores(self):
        """
        Generate final scores using adjudication if available, otherwise average.
        
        Returns:
            Self for method chaining
        """
        if not self.evaluations:
            raise ValueError("No evaluations found. Cannot generate final scores.")
            
        final_scores = {}
        incomplete_sessions = [] # Sessions with < 2 evaluations
        processing_errors = [] # Sessions with processing errors
        
        for session_id, evals in self.evaluations.items():
            try:
                # Check if adjudication exists (3rd evaluation)
                if len(evals) >= 3:
                    final_eval = evals[2][0]
                
                # Average two evaluations
                elif len(evals) == 2:
                    eval1, eval2 = evals[0][0], evals[1][0]
                    
                    # Validate structure
                    if not all(isinstance(e.get('scores'), dict) for e in [eval1, eval2]):
                        incomplete_sessions.append({
                            'session_id': session_id,
                            'eval_count': len(evals),
                            'issue': 'Missing or invalid scores field'
                        })
                        continue
                    
                    # Check if structures match
                    # Check categories and subcategories
                    if (set(eval1['scores'].keys()) != set(eval2['scores'].keys()) or
                        any(set(eval1['scores'][cat].keys()) != set(eval2['scores'][cat].keys()) 
                            for cat in eval1['scores'])):
                        incomplete_sessions.append({
                            'session_id': session_id,
                            'eval_count': len(evals),
                            'issue': 'Mismatched evaluation structure'
                        })
                        continue
                    
                    # Start with copy of eval1 to preserve all fields
                    final_eval = eval1.copy()
                    final_eval['scores'] = {}
                    
                    # Average scores
                    for category in eval1['scores']:
                        final_eval['scores'][category] = {}
                        for subcategory in eval1['scores'][category]:
                            score1 = eval1['scores'][category][subcategory]
                            score2 = eval2['scores'][category][subcategory]

                            # Coerce string scores to numeric (handles legacy pkl data)
                            if isinstance(score1, str):
                                try: score1 = int(score1)
                                except ValueError:
                                    try: score1 = float(score1)
                                    except ValueError: pass
                            if isinstance(score2, str):
                                try: score2 = int(score2)
                                except ValueError:
                                    try: score2 = float(score2)
                                    except ValueError: pass

                            if score1 is None and score2 is None:
                                final_eval['scores'][category][subcategory] = None
                            elif score1 is None or score2 is None:
                                final_eval['scores'][category][subcategory] = score1 if score1 is not None else score2
                            else:
                                final_eval['scores'][category][subcategory] = round((score1 + score2) / 2)
                
                else:
                    # Incomplete sessions (0 or 1 evaluation)
                    incomplete_sessions.append({
                        'session_id': session_id,
                        'eval_count': len(evals)
                    })
                    continue
                
                final_scores[session_id] = final_eval
                
            except Exception as e:
                processing_errors.append({
                    'session_id': session_id,
                    'error': str(e)
                })
                self.logger.error(f"Error processing final score for session {session_id}: {e}")
                continue
        
        # Save results
        self.config.dirs.evaluation_results.mkdir(parents=True, exist_ok=True)
        
        output_path = self.config.dirs.evaluation_results / f"{self.config.run_id}_final_scores.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_scores, f, indent=2)
        
        if incomplete_sessions:
            incomplete_path = self.config.dirs.evaluation_results / f"{self.config.run_id}_incomplete_sessions.json"
            with open(incomplete_path, 'w', encoding='utf-8') as f:
                json.dump(incomplete_sessions, f, indent=2)
            self.logger.warning(f"{len(incomplete_sessions)} session(s) incomplete - saved to {incomplete_path}")
        
        if processing_errors:
            errors_path = self.config.dirs.evaluation_results / f"{self.config.run_id}_processing_errors.json"
            with open(errors_path, 'w', encoding='utf-8') as f:
                json.dump(processing_errors, f, indent=2)
            self.logger.warning(f"Processing errors saved to {errors_path}")
        
        self.logger.info(f"Final scores saved to {output_path}")
        self.logger.info(f"Total sessions scored: {len(final_scores)}")
        
        if incomplete_sessions:
            self.logger.info(f"  {len(incomplete_sessions)} sessions excluded due to incomplete evaluations")
        
        return self

    # ========================================================================
    # STATUS AND DIAGNOSTICS
    # ========================================================================

    def check_evaluation_status(self) -> Dict[str, Any]:
        """
        Check the status of evaluations across all sessions.
        
        Args:
            verbose: If True, print a formatted summary
            
        Returns:
            Dictionary with evaluation status information
        """
        if self.session_data is None:
            self.logger.warning("Session data not loaded. Cannot check evaluation status.")
            return {}
        
        # Initialize status dict
        status = {
            'total_sessions': len(self.session_data),
            'not_started': [],
            'in_progress': [],
            'complete': [],
            'needs_adjudication': [],
            'ready_for_final_scores': [],
            'has_dynamic_prompts': len(self.dynamic_prompts) > 0,
            'batch_file_created': self.batch_file_path is not None and self.batch_file_path.exists(),
            'batch_uploaded': self.batch_id is not None
        }

        # Categorize each session
        for session_id in self.session_data['session_id']:
            eval_count = len(self.evaluations.get(session_id, []))
            
            if eval_count == 0:
                status['not_started'].append(session_id)
            elif eval_count == 1:
                status['in_progress'].append(session_id)
            elif eval_count >= 2:
                status['complete'].append(session_id)
                
                # Check if needs adjudication
                needs_adj, reason = utils.needs_adjudication(
                    self.evaluations[session_id][0][0],
                    self.evaluations[session_id][1][0]
                )
                
                if needs_adj:
                    if eval_count < 3:
                        status['needs_adjudication'].append({
                            'session_id': session_id,
                            'reason': reason
                        })
                    else:
                        status['ready_for_final_scores'].append(session_id)
                else:
                    status['ready_for_final_scores'].append(session_id)
        
        # Determine next action
        if status['not_started']:
            if status['has_dynamic_prompts']:
                if status['batch_file_created'] and not status['batch_uploaded']:
                    next_action = "Run upload_batch() to submit batch for processing"
                elif status['batch_uploaded']:
                    next_action = "Run check_and_retrieve() to check batch status and retrieve results"
                else:
                    next_action = f"Run flex_evaluate() OR prepare_batch_file() to evaluate {len(status['not_started'])} sessions"
            else:
                next_action = f"Run generate_dynamic_prompts() to create prompts for {len(status['not_started'])} sessions"
        elif status['in_progress']:
            next_action = f"Continue evaluations for {len(status['in_progress'])} sessions"
        elif status['needs_adjudication']:
            if status['has_dynamic_prompts']:
                if status['batch_file_created'] and not status['batch_uploaded']:
                    next_action = "Run upload_batch() for adjudication batch"
                elif status['batch_uploaded']:
                    next_action = "Run check_and_retrieve() for adjudication results"
                else:
                    next_action = f"Run flex_evaluate(adjudication=True) OR prepare_batch_file(adjudication=True) for {len(status['needs_adjudication'])} sessions"
            else:
                next_action = f"Run generate_dynamic_prompts(adjudication=True) for {len(status['needs_adjudication'])} sessions"
        elif status['ready_for_final_scores']:
            next_action = f"Run generate_final_scores() to complete the pipeline if you haven't already; check {self.config.dirs.evaluation_results} for existing final scores"
        else:
            next_action = "No action needed"
        
        status['next_action'] = next_action

        return status

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
        import pickle

        results = []

        if all_runs:
            pkl_files = sorted(self.config.dirs.evaluation_results.glob("*_evaluations.pkl"))
        else:
            pkl_files = [self.config.dirs.evaluation_results / f"{self.config.run_id}_evaluations.pkl"]
            pkl_files = [p for p in pkl_files if p.exists()]

        if not pkl_files and self.evaluations:
            # Use in-memory evaluations
            pkl_files = []
            evals_to_scan = [("current", self.evaluations)]
        else:
            evals_to_scan = []
            for p in pkl_files:
                with open(p, "rb") as f:
                    evals_to_scan.append((p.stem.replace("_evaluations", ""), pickle.load(f)))

        grand_input = 0
        grand_output = 0
        grand_cached = 0
        grand_evals = 0

        print(f"{'Run ID':<20} {'Sessions':>8} {'Evals':>6} {'Input Tok':>12} {'Output Tok':>12} {'Cost':>10}")
        print("=" * 78)

        for run_id, evals in evals_to_scan:
            n_sessions = len(evals)
            n_evals = sum(len(v) for v in evals.values())
            total_input = 0
            total_output = 0
            total_cached = 0

            for sid, runs in evals.items():
                for parsed_eval, response in runs:
                    if hasattr(response, "usage") and response.usage:
                        u = response.usage
                        total_input += u.input_tokens
                        total_output += u.output_tokens
                        if hasattr(u, "input_tokens_details") and u.input_tokens_details:
                            total_cached += getattr(u.input_tokens_details, "cached_tokens", 0)

            uncached = total_input - total_cached
            cost = (
                uncached * self.config.model.input_token_price
                + total_cached * self.config.model.cached_token_price
                + total_output * self.config.model.output_token_price
            )

            print(f"{run_id:<20} {n_sessions:>8} {n_evals:>6} {total_input:>12,} {total_output:>12,} {'$' + f'{cost:.2f}':>10}")

            grand_input += total_input
            grand_output += total_output
            grand_cached += total_cached
            grand_evals += n_evals

            results.append({
                "run_id": run_id,
                "sessions": n_sessions,
                "evaluations": n_evals,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cached_tokens": total_cached,
                "cost": cost,
            })

        if len(evals_to_scan) > 1:
            uncached_grand = grand_input - grand_cached
            grand_cost = (
                uncached_grand * self.config.model.input_token_price
                + grand_cached * self.config.model.cached_token_price
                + grand_output * self.config.model.output_token_price
            )
            print("=" * 78)
            print(f"{'TOTAL':<20} {'':>8} {grand_evals:>6} {grand_input:>12,} {grand_output:>12,} {'$' + f'{grand_cost:.2f}':>10}")
            print(f"\nCached input tokens: {grand_cached:,} ({grand_cached/grand_input*100:.0f}%)" if grand_input else "")

        return results

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
        if self.rag_dictionary is None or self.rag_embeddings is None:
            raise ValueError("RAG data not loaded.")

        if not self.guidelines.get("guideline_final.txt"):
            raise ValueError("Guidelines not generated.")

        final_guideline = self.guidelines["guideline_final.txt"]
        n_few_shot = self.config.evaluation_settings.n_human_rating_samples

        self.dynamic_prompts = {}

        for _, row in tqdm(sniff_data.iterrows(), total=len(sniff_data), desc="Generating sniff test prompts"):
            session_id = str(row["session_id"])

            # --- Few-shot pool: exclude current session ---
            if self.human_evaluation is not None and len(self.human_evaluation) > 0:
                pool = self.human_evaluation[
                    self.human_evaluation["session_id"].astype(str) != session_id
                ]
                # Sample up to n_few_shot from the remaining pool
                n_sample = min(n_few_shot, len(pool))
                if n_sample > 0:
                    few_shot_samples = pool.sample(n=n_sample, random_state=42)
                else:
                    few_shot_samples = None
            else:
                few_shot_samples = None

            # --- RAG context ---
            row_string = utils.format_any_tabular_data(
                row[~row.index.isin(["image_data_base64"])],
                f"{self.config.tool_settings.tool_name} Data",
            )
            row_string = utils.truncate_text_to_tokens(row_string, max_tokens=8000)
            embedding = self.client.create_embedding(row_string)
            similar_entries = retrieve_similar(embedding, self.rag_embeddings, top_k=1)
            similar_cid = similar_entries[0][0]
            rag_context = self.rag_dictionary.get(similar_cid, "No similar content found.")

            # --- Build prompt with few-shot examples ---
            prompt = self.prompt_builder.build_evaluation_prompt(
                row=row,
                rag_context=rag_context,
                guideline=final_guideline,
                human_evaluation_samples=few_shot_samples,
            )

            self.dynamic_prompts[session_id] = prompt

        self.logger.info(f"Generated {len(self.dynamic_prompts)} sniff test prompts (with per-session few-shot exclusion)")

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
        from . import error_analysis
        from jinja2 import Template

        if self.human_evaluation is None:
            raise ValueError(
                "Human evaluation data required for sniff test. "
                "Ensure human_evaluation path is set in config and the file exists."
            )

        if "guideline_final.txt" not in self.guidelines:
            raise ValueError(
                "Guidelines must be generated before running sniff test. "
                "Call generate_evaluation_guidelines() first."
            )

        # Load the FULL human evaluation CSV (unsampled) for sniff test comparison.
        # self.human_evaluation may be sampled via n_human_rating_samples (used for
        # guideline generation few-shot), but the sniff test needs all human reviews.
        full_human_eval = pd.read_csv(self.config.file_paths.human_evaluation)
        if 'image_data_base64' in full_human_eval.columns:
            full_human_eval = full_human_eval.drop(columns=['image_data_base64'])

        # --- Step 1: Load ALL human-reviewed sessions from the full CSV ---
        # We bypass self.session_data (which is sampled via n_samples) and load
        # the complete session CSV so the sniff test covers all human-reviewed sessions.
        from . import data as data_module
        from ast import literal_eval

        full_session_data = pd.read_csv(self.config.file_paths.session_data)
        if self.data_prep_function:
            full_session_data = self.data_prep_function(full_session_data, self.config)
        if "image_data_base64" in full_session_data.columns and full_session_data["image_data_base64"].dtype == object:
            full_session_data["image_data_base64"] = full_session_data["image_data_base64"].apply(
                lambda x: literal_eval(x) if pd.notna(x) else None
            )

        human_session_ids = set(full_human_eval[human_id_col].astype(str))

        sniff_data = full_session_data[
            full_session_data["session_id"].astype(str).isin(human_session_ids)
        ]

        if len(sniff_data) == 0:
            raise ValueError(
                f"No overlap between human evaluation session IDs and session data. "
                f"Human IDs sample: {list(human_session_ids)[:3]}, "
                f"Session IDs sample: {list(full_session_data['session_id'].astype(str))[:3]}"
            )

        self.logger.info(
            f"Sniff test: {len(sniff_data)} sessions with human review data "
            f"(out of {len(human_session_ids)} human-reviewed)"
        )

        # --- Step 2: Evaluate with per-session few-shot exclusion ---
        original_session_data = self.session_data
        original_evaluations = self.evaluations.copy()
        original_dynamic_prompts = self.dynamic_prompts.copy()

        self.session_data = sniff_data.reset_index(drop=True)

        sniff_session_ids = set(sniff_data["session_id"].astype(str))
        for sid in sniff_session_ids:
            if sid in self.evaluations:
                del self.evaluations[sid]

        self.dynamic_prompts = {}

        try:
            # Use sniff-test-specific prompt generation (with few-shot exclusion)
            self._generate_sniff_test_prompts(self.session_data)
            self.flex_evaluate(
                service_tier=service_tier,
                adjudication=False,
                n_runs=2,
                auto_approve=auto_approve,
            )

            self.generate_final_scores()

            scores_path = self.config.dirs.evaluation_results / f"{self.config.run_id}_final_scores.json"
            with open(scores_path, "r", encoding="utf-8") as f:
                llm_scores = json.load(f)

        finally:
            self.session_data = original_session_data
            for sid in sniff_session_ids:
                if sid in self.evaluations:
                    original_evaluations[sid] = self.evaluations[sid]
            self.evaluations = original_evaluations
            self.dynamic_prompts = original_dynamic_prompts

        # --- Step 3: Compare scores ---
        comparison_df = error_analysis.compare_scores(
            llm_scores=llm_scores,
            human_scores_df=full_human_eval,
            session_id_col=human_id_col,
            column_map=column_map,
        )

        if comparison_df.empty:
            self.logger.warning("No score comparisons could be made. Check ID matching.")
            return {"comparison_df": comparison_df, "metrics": {}, "llm_analysis": "", "report": ""}

        metrics = error_analysis.compute_agreement_metrics(comparison_df)

        self.logger.info(
            f"Sniff test comparison: {metrics['overall'].get('n_comparisons', 0)} score pairs, "
            f"{metrics['overall'].get('exact_match_pct', 0)}% exact match, "
            f"bias: {metrics['overall'].get('bias_direction', 'N/A')}"
        )

        # --- Step 4: LLM error analysis ---
        error_analysis_template_path = Path("inputs/prompts/error_analysis.j2")
        if not error_analysis_template_path.exists():
            self.logger.warning("error_analysis.j2 template not found, skipping LLM analysis")
            llm_analysis = "(Template not found — skipping LLM-assisted analysis)"
        else:
            template = Template(error_analysis_template_path.read_text(encoding="utf-8"))

            llm_explanations = ""
            worst_sessions = (
                comparison_df.groupby("session_id")["abs_difference"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .index
            )
            for sid in worst_sessions:
                if sid in llm_scores:
                    explanations = llm_scores[sid].get("explanations", {})
                    if explanations:
                        llm_explanations += f"\n### Session {sid[:12]}...\n"
                        llm_explanations += json.dumps(explanations, indent=2) + "\n"

            prompt = template.render(
                n_sessions=len(sniff_data),
                guidelines=self.guidelines.get("guideline_final.txt", ""),
                rubric=self.prompt_builder.rubric_json,
                tool_description=self.prompt_builder.tool_description,
                metrics_summary=error_analysis.format_metrics_summary(metrics),
                comparison_table=error_analysis.format_comparison_table(comparison_df),
                llm_explanations=llm_explanations if llm_explanations else None,
            )

            self.logger.info("Running LLM error analysis...")
            response = self.client.call(prompt, service_tier=service_tier)
            llm_analysis = response.output_text.strip()

        # --- Step 5: Generate and save report ---
        report = error_analysis.format_error_analysis_report(
            comparison_df, metrics, llm_analysis
        )

        report_path = (
            self.config.dirs.evaluation_results
            / f"{self.config.run_id}_sniff_test_report.md"
        )
        report_path.write_text(report, encoding="utf-8")
        self.logger.info(f"Sniff test report saved to {report_path}")

        return {
            "comparison_df": comparison_df,
            "metrics": metrics,
            "llm_analysis": llm_analysis,
            "report": report,
        }

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
        current_guidelines = self.guidelines.get("guideline_final.txt", "")
        if not current_guidelines:
            raise ValueError("No current guidelines found. Generate guidelines first.")

        prompt = (
            "You are an expert in educational assessment and evaluation guideline design.\n\n"
            "Below are the CURRENT evaluation guidelines used by an LLM judge to evaluate "
            "AI tutoring sessions, followed by an ERROR ANALYSIS REPORT comparing the LLM "
            "judge's scores against human expert scores on a small calibration set.\n\n"
            "Your task: Produce a REVISED version of the evaluation guidelines that "
            "incorporates the insights from the error analysis. Specifically:\n"
            "- Address systematic biases identified in the report\n"
            "- Clarify scoring anchors for criteria where human-LLM disagreement was highest\n"
            "- Add calibration notes for criteria the report flagged as problematic\n"
            "- Preserve all existing guideline structure and content that was NOT flagged\n\n"
            "CRITICAL: Output the COMPLETE revised guidelines text from start to finish. "
            "Do NOT use placeholders like '[Section text unchanged]' or '[rest remains the same]' — "
            "every section must be written out in full, even if unchanged. "
            "No preamble, no meta-commentary.\n\n"
            "---\n\n"
            "## CURRENT EVALUATION GUIDELINES\n\n"
            f"{current_guidelines}\n\n"
            "---\n\n"
            "## ERROR ANALYSIS REPORT\n\n"
            f"{sniff_test_report}\n"
        )

        self.logger.info("Refining guidelines based on sniff test report...")
        response = self.client.call(prompt, service_tier=service_tier)
        refined = response.output_text.strip()

        # Save refined guidelines
        refined_path = self.config.dirs.evaluation_guidelines / guideline_name
        refined_path.write_text(refined, encoding="utf-8")
        self.guidelines[guideline_name] = refined
        self.logger.info(f"Refined guidelines saved to {refined_path}")

        return refined

    def use_refined_guidelines(self, guideline_name: str = "guideline_refined.txt"):
        """
        Swap the active final guideline to the refined version.

        This copies the refined guideline over guideline_final.txt so subsequent
        evaluation steps use the refined version.

        Args:
            guideline_name: Name of the refined guideline file to activate
        """
        if guideline_name not in self.guidelines:
            raise ValueError(
                f"Guideline '{guideline_name}' not found. "
                f"Available: {list(self.guidelines.keys())}"
            )

        refined_text = self.guidelines[guideline_name]

        # Back up current final guideline
        backup_name = "guideline_final_pre_refinement.txt"
        if "guideline_final.txt" in self.guidelines and backup_name not in self.guidelines:
            backup_path = self.config.dirs.evaluation_guidelines / backup_name
            backup_path.write_text(self.guidelines["guideline_final.txt"], encoding="utf-8")
            self.guidelines[backup_name] = self.guidelines["guideline_final.txt"]
            self.logger.info(f"Backed up original guidelines to {backup_name}")

        # Overwrite final guideline
        self.guidelines["guideline_final.txt"] = refined_text
        final_path = self.config.dirs.evaluation_guidelines / "guideline_final.txt"
        final_path.write_text(refined_text, encoding="utf-8")
        self.logger.info(f"Active guidelines updated from {guideline_name}")

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