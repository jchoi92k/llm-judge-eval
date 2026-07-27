# evaluation_pipeline/guidelines.py

"""
Guideline generation: multi-run LLM guideline drafting + final aggregation.

Plain functions operating on an Evaluator instance (delegation pattern,
same as data.py). The Evaluator methods are one-line delegates to these.
"""

from pathlib import Path

from . import utils


def aggregate_texts(path: Path) -> str:
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


def generate_evaluation_guidelines(evaluator, service_tier: str = "flex", n_runs: int = 2, auto_approve: bool = False, force_regenerate: bool = False, test_run: bool = False):
    """Generate evaluation guidelines using multiple LLM runs + aggregation."""
    # Define guideline file names (shared across runs)
    # Guideline names are hardcoded
    guideline_file_names = [f"guideline_run_{i+1}.txt" for i in range(n_runs)]
    guideline_file_names.append("guideline_final.txt")

    # Check if final guideline already exists
    final_guideline_path = evaluator.config.dirs.evaluation_guidelines / "guideline_final.txt"

    if final_guideline_path.exists() and not force_regenerate:
        evaluator.logger.info("Existing guidelines found. Loading from file.")
        evaluator.logger.info("Guidelines are not tied to specific runs. To regenerate, use force_regenerate=True or delete guideline files.")

        # Load existing guidelines
        for file_name in guideline_file_names:
            file_path = evaluator.config.dirs.evaluation_guidelines / file_name
            if file_path.exists():
                evaluator.guidelines[file_name] = file_path.read_text(encoding='utf-8')

        return evaluator

    # If force_regenerate, clear existing guidelines
    if force_regenerate:
        try:
            evaluator.logger.info("Force regeneration requested. Deleting existing guidelines.")
            for file_name in guideline_file_names:
                file_path = evaluator.config.dirs.evaluation_guidelines / file_name
                if file_path.exists():
                    file_path.unlink()
            evaluator.guidelines.clear()
        # If error due to file open etc., log and raise
        except Exception as e:
            evaluator.logger.error(f"Failed to delete existing guidelines: {e}")
            raise

    # Load existing guidelines (for partial regeneration)
    for file_name in guideline_file_names:
        file_path = evaluator.config.dirs.evaluation_guidelines / file_name
        if file_path.exists():
            evaluator.guidelines[file_name] = file_path.read_text(encoding='utf-8')

    # If all exist, skip generation
    if len(evaluator.guidelines) == len(guideline_file_names):
        evaluator.logger.info("All guidelines already exist, skipping generation")
        return evaluator

    # Aggregate practice guides
    if test_run:
        practice_guides = "Test practice guide content."
    else:
        practice_guides = aggregate_texts(evaluator.config.dirs.practice_guides)

    # Build guideline prompt
    if evaluator.human_evaluation is None:
        if not auto_approve:
            utils.flush_logs()
            user_input = input("Human evaluation data not loaded. Proceed without human evaluation? (y/n): ")
            if user_input.lower() != 'y':
                evaluator.logger.info("Guideline generation cancelled")
                return evaluator

        # use n=1 samples of session data to show sample data sans human evaluations
        sample_data = evaluator.session_data.sample(n=1, random_state=evaluator.config.evaluation_settings.random_state).reset_index(drop=True)

        # remove image data from sample data for now
        if 'image_data_base64' in sample_data.columns:
            sample_data = sample_data.drop(columns=['image_data_base64'])

        guideline_prompt = evaluator.prompt_builder.build_guideline_prompt(sample_data, practice_guides)
    else:
        guideline_prompt = evaluator.prompt_builder.build_guideline_prompt(evaluator.human_evaluation, practice_guides)

    # TEST LINE
    evaluator.test_components.append(guideline_prompt)

    # Rough cost
    guideline_run_1 = evaluator.client.estimate_cost(
        prompt_cached="",
        prompt_uncached=guideline_prompt,
        expected_output_tokens=evaluator.config.model.expected_output_tokens
    )

    guideline_run_2 = evaluator.client.estimate_cost(
        prompt_cached=guideline_prompt,
        prompt_uncached="",
        expected_output_tokens=evaluator.config.model.expected_output_tokens
    )

    guideline_run_adjudcation = evaluator.client.estimate_cost(
        prompt_cached="",
        prompt_uncached=guideline_prompt,
        expected_output_tokens=evaluator.config.model.expected_output_tokens
    )

    guideline_cost = guideline_run_1 + guideline_run_2 + guideline_run_adjudcation

    evaluator.logger.info(f"Estimated guideline generation cost: ${guideline_cost:.4f}")
    utils.flush_logs()

    # Get approval
    if not auto_approve:
        user_input = input("Proceed with guideline generation? (y/n): ")
        if user_input.lower() != 'y':
            evaluator.logger.info("Guideline generation cancelled")
            return evaluator

    # Generate guidelines
    for i in range(n_runs):
        file_name = f"guideline_run_{i+1}.txt"
        if file_name in evaluator.guidelines:
            continue

        evaluator.logger.info(f"Generating guideline {i+1}/{n_runs}...")
        response = evaluator.client.call(guideline_prompt, service_tier=service_tier)
        guideline_text = response.output_text.strip()

        evaluator.guidelines[file_name] = guideline_text

        # Save to disk
        file_path = evaluator.config.dirs.evaluation_guidelines / file_name
        file_path.write_text(guideline_text, encoding='utf-8')

    # Generate final aggregated guideline
    final_file_name = "guideline_final.txt"
    if final_file_name not in evaluator.guidelines:
        evaluator.logger.info("Generating final aggregated guideline...")

        guideline_outputs = [v for k, v in evaluator.guidelines.items() if "guideline_final" not in k]

        adjudication_prompt = evaluator.prompt_builder.build_guidelines_aggregation_prompt(
            original_prompt=guideline_prompt,
            guideline_outputs=guideline_outputs,
        )

        response = evaluator.client.call(adjudication_prompt, service_tier=service_tier)
        final_guideline = response.output_text.strip()

        evaluator.guidelines[final_file_name] = final_guideline
        file_path = evaluator.config.dirs.evaluation_guidelines / final_file_name
        file_path.write_text(final_guideline, encoding='utf-8')

        # TEST LINE
        evaluator.test_components.append(adjudication_prompt)

    evaluator.logger.info("Guideline generation complete")
    return evaluator
