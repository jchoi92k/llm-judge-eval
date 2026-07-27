# evaluation_pipeline/execution.py

"""
Evaluation execution: direct (flex) API calls and OpenAI batch API processing.

Plain functions operating on an Evaluator instance (delegation pattern,
same as data.py). The Evaluator methods are one-line delegates to these.
"""

import json
import pickle
import time
from typing import Optional

from tqdm import tqdm

from . import utils


# ============================================================================
# FLEX EVALUATION (Direct API calls)
# ============================================================================

def _call_and_record(evaluator, session_id, prompt, service_tier, context):
    """
    Run one evaluation call: call the API, parse the output, append to
    evaluator.evaluations. Shared by the adjudication and normal flex paths.

    Args:
        context: Label for error messages, e.g. "adjudication" or "evaluation run 1/2".

    Returns:
        True if the evaluation was parsed and recorded, False if parsing failed.
    """
    response = evaluator.client.call(prompt, service_tier=service_tier)
    eval_text = response.output_text.strip()

    success, parsed_eval, error_msg = utils.try_parse_evaluation(eval_text)

    if not success:
        evaluator.logger.error(f"Failed to parse {context} for {session_id}: {error_msg}")
        return False

    evaluator.evaluations[session_id].append([parsed_eval, response])
    return True


def flex_evaluate(evaluator, service_tier: str = "flex", adjudication: bool = False, n_runs: int = 2, auto_approve: bool = False):
    """Run evaluations using direct API calls with flex pricing."""
    if not evaluator.dynamic_prompts:
        raise ValueError("No dynamic prompts generated. Call generate_dynamic_prompts() first. Or you might be in adjudication mode with no sessions needing adjudication.")

    # Find common cached prefix
    all_prompts = list(evaluator.dynamic_prompts.values())
    if all_prompts:

        cached_prefix, uncached_text = utils.find_prefix(all_prompts)

        text_cost_per_eval = evaluator.client.estimate_cost(
            prompt_cached=cached_prefix,
            prompt_uncached=uncached_text,
            expected_output_tokens=500  # evaluation output estimate hardcoded for now
        )

        # Estimate image token cost
        image_counts = [utils.count_images_in_prompt(p) for p in all_prompts]
        avg_images = sum(image_counts) / len(image_counts) if image_counts else 0
        avg_image_tokens = utils.estimate_image_tokens(int(avg_images))
        image_cost_per_eval = avg_image_tokens * evaluator.config.model.input_token_price

        cost_per_evaluation = text_cost_per_eval + image_cost_per_eval
        total_cost = cost_per_evaluation * len(evaluator.dynamic_prompts) * n_runs

        n_with_images = sum(1 for c in image_counts if c > 0)
        evaluator.logger.info(
            f"Estimated evaluation cost: ${total_cost:.4f} "
            f"(text: ${text_cost_per_eval * len(evaluator.dynamic_prompts) * n_runs:.4f}, "
            f"images: ${image_cost_per_eval * len(evaluator.dynamic_prompts) * n_runs:.4f} — "
            f"{n_with_images}/{len(all_prompts)} prompts with images, avg {avg_images:.1f} images/prompt)"
        )

        if not auto_approve:
            utils.flush_logs()
            user_input = input("Proceed with evaluation? (y/n): ")
            if user_input.lower() != 'y':
                evaluator.logger.info("Evaluation cancelled")
                return evaluator

    if adjudication:
        n_runs = 1
        evaluator.logger.info("Adjudication mode: using 1 run per session")
    else:
        evaluator.logger.info(f"Evaluation mode: using {n_runs} runs per session")

    # Run evaluations
    for session_id, prompt in tqdm(evaluator.dynamic_prompts.items(), desc="Evaluating sessions"):
        if adjudication:
            # Adjudication mode: always run exactly 1 evaluation
            evaluator.logger.debug(f"Running adjudication for {session_id}")
            if not _call_and_record(evaluator, session_id, prompt, service_tier, "adjudication"):
                continue  # parse failure: skip the per-session save, matching pre-refactor flow

        else:
            # Regular evaluation mode: run n_runs evaluations
            # Skip if already have enough evaluations
            if len(evaluator.evaluations[session_id]) >= n_runs:
                continue

            for run in range(n_runs):
                # Skip if we already have this run
                if len(evaluator.evaluations[session_id]) > run:
                    continue

                evaluator.logger.debug(f"Evaluating {session_id}, run {run + 1}/{n_runs}")
                _call_and_record(evaluator, session_id, prompt, service_tier, f"evaluation run {run + 1}/{n_runs}")

        # Save after each session
        evaluator._save_evaluations()

    evaluator.logger.info("Flex evaluation complete")
    return evaluator


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def _mapping_path_for(batch_file_path):
    """Sidecar JSON file (custom_id → session_id) next to a batch JSONL file."""
    return batch_file_path.with_name(batch_file_path.stem + "_mapping.json")


def prepare_batch_file(evaluator, adjudication: bool = False, n_runs: int = 2, auto_approve: bool = False):
    """Prepare JSONL batch file for OpenAI batch API."""
    if not evaluator.dynamic_prompts:
        raise ValueError("No dynamic prompts generated. Call generate_dynamic_prompts() first.")

    timestamp = int(time.time())
    evaluator.batch_file_path = evaluator.config.dirs.batch_processing / f"{evaluator.config.run_id}_{timestamp}_batch.jsonl"

    if evaluator.batch_file_path.exists():
        evaluator.logger.info("Batch file already exists, skipping creation")
        return evaluator

    # Find common cached prefix
    all_prompts = list(evaluator.dynamic_prompts.values())
    if all_prompts:
        cached_prefix, uncached_text = utils.find_prefix(all_prompts)

        text_cost_per_eval = evaluator.client.estimate_cost(
            prompt_cached=cached_prefix,
            prompt_uncached=uncached_text,
            expected_output_tokens=500  # evaluation output estimate hardcoded for now
        )

        # Estimate image token cost
        image_counts = [utils.count_images_in_prompt(p) for p in all_prompts]
        avg_images = sum(image_counts) / len(image_counts) if image_counts else 0
        avg_image_tokens = utils.estimate_image_tokens(int(avg_images))
        image_cost_per_eval = avg_image_tokens * evaluator.config.model.input_token_price

        cost_per_evaluation = text_cost_per_eval + image_cost_per_eval
        total_cost = cost_per_evaluation * len(evaluator.dynamic_prompts) * n_runs

        n_with_images = sum(1 for c in image_counts if c > 0)
        evaluator.logger.info(
            f"Estimated batch processing cost: ${total_cost:.4f} "
            f"(text: ${text_cost_per_eval * len(evaluator.dynamic_prompts) * n_runs:.4f}, "
            f"images: ${image_cost_per_eval * len(evaluator.dynamic_prompts) * n_runs:.4f} — "
            f"{n_with_images}/{len(all_prompts)} prompts with images, avg {avg_images:.1f} images/prompt)"
        )

        if not auto_approve:
            utils.flush_logs()
            user_input = input("Proceed with batch file creation? (y/n): ")
            if user_input.lower() != 'y':
                evaluator.logger.info("Batch file creation cancelled")
                return evaluator

    request_counter = 0

    if adjudication:
        n_runs = 1
        evaluator.logger.info("Adjudication mode: using 1 run per session")
    else:
        evaluator.logger.info(f"Evaluation mode: using {n_runs} runs per session")

    custom_id_map = {}

    for session_id, prompt in evaluator.dynamic_prompts.items():
        for _ in range(n_runs):
            custom_id = f"{evaluator.config.run_id}_{timestamp}_{request_counter}"
            custom_id_map[custom_id] = session_id

            batch_entry = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": evaluator.config.model.model_name,
                    "input": prompt
                }
            }

            with open(evaluator.batch_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(batch_entry) + "\n")

            request_counter += 1

    # Sidecar mapping so retrieval can resolve sessions directly instead of
    # inferring them from request order (which breaks for adjudication batches)
    mapping_path = _mapping_path_for(evaluator.batch_file_path)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(custom_id_map, f, indent=2)

    evaluator.logger.info(f"Created batch file at {evaluator.batch_file_path} with {request_counter} requests")
    evaluator.logger.info(f"Wrote custom_id → session mapping to {mapping_path}")
    return evaluator


def upload_batch(evaluator):
    """Upload batch file to OpenAI and create batch job."""
    if not evaluator.batch_file_path or not evaluator.batch_file_path.exists():
        raise ValueError("Batch file not found. Call prepare_batch_file() first.")

    evaluator.batch_id = evaluator.client.upload_batch_file(evaluator.batch_file_path)
    evaluator.logger.info(f"Batch uploaded with ID: {evaluator.batch_id}")
    return evaluator


def check_batch_status(evaluator) -> str:
    """Check the status of the current batch job."""
    if not evaluator.batch_id:
        raise ValueError("No batch ID found. Call upload_batch() first.")

    info = evaluator.client.check_batch_status(evaluator.batch_id)
    status = info["status"]
    evaluator.logger.info(f"Batch status: {status} ({info['completed']}/{info['total']} completed, {info['failed']} failed)")
    return status


def check_and_retrieve(evaluator, until_complete: bool = False, check_interval: int = 60, batch_id_override: Optional[str] = None):
    """Check batch status and optionally wait until complete + retrieve results."""
    # Use override if provided, otherwise use evaluator.batch_id
    batch_id = batch_id_override or evaluator.batch_id

    if not batch_id:
        raise ValueError("No batch ID found. Call upload_batch() first or provide batch_id_override.")

    if not until_complete:
        info = evaluator.client.check_batch_status(batch_id)
        evaluator.logger.info(f"Batch status: {info['status']} ({info['completed']}/{info['total']} completed, {info['failed']} failed)")
        return evaluator

    # Wait until complete
    while True:
        info = evaluator.client.check_batch_status(batch_id)
        status = info["status"]
        evaluator.logger.info(f"Batch status: {status} ({info['completed']}/{info['total']} completed, {info['failed']} failed)")

        if status == "completed":
            evaluator.logger.info("Batch completed successfully")
            retrieve_batch_results(evaluator, batch_id)
            break
        elif status in ["failed", "cancelled"]:
            evaluator.logger.error(f"Batch ended with status: {status}")
            break
        else:
            evaluator.logger.info(f"Checking again in {check_interval} seconds...")
            time.sleep(check_interval)

    return evaluator


def retrieve_batch_results(evaluator, batch_id_override: Optional[str] = None):
    """Retrieve and process batch results."""
    batch_id = batch_id_override or evaluator.batch_id

    if not batch_id:
        raise ValueError("No batch ID found")

    evaluator.logger.info("Retrieving batch results...")
    results = evaluator.client.retrieve_batch_results(batch_id)

    # Save raw results
    output_filename = evaluator.config.dirs.batch_processing_results / f"batch_results_{evaluator.config.run_id}.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump({'results': results}, f)

    # Load the custom_id → session_id sidecar mapping written by prepare_batch_file.
    # Legacy batches (created before the mapping existed) fall back to inferring
    # the session from request order, which assumes 2 runs per session.
    custom_id_map = {}
    if evaluator.batch_file_path is not None:
        mapping_path = _mapping_path_for(evaluator.batch_file_path)
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                custom_id_map = json.load(f)
    if not custom_id_map:
        evaluator.logger.warning(
            "No custom_id mapping file found; falling back to request-order mapping "
            "(assumes 2 runs per session — WRONG for adjudication batches)"
        )

    # Process results into evaluations
    session_ids = list(evaluator.dynamic_prompts.keys())
    n_runs = 2 # Legacy fallback only: assumes 2 runs per session
    processed_count = 0
    failed_count = 0

    for result in results:
        try:
            custom_id = result.get('custom_id', '')
            if not custom_id:
                evaluator.logger.warning("Missing custom_id in batch result")
                failed_count += 1
                continue

            if custom_id_map:
                session_id = custom_id_map.get(custom_id)
                if session_id is None:
                    evaluator.logger.warning(f"custom_id {custom_id} not found in batch mapping")
                    failed_count += 1
                    continue
            else:
                counter = int(custom_id.split('_')[-1])
                session_idx = counter // n_runs

                if session_idx >= len(session_ids):
                    evaluator.logger.warning(f"Session index {session_idx} out of range (max: {len(session_ids)-1})")
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
                    evaluator.evaluations[session_id].append([parsed_eval, result['response']])
                    processed_count += 1
                else:
                    evaluator.logger.error(f"Failed to parse evaluation for {session_id}: {error_msg}")
                    failed_count += 1
            else:
                evaluator.logger.warning(f"No evaluation text found for session {session_id}")
                failed_count += 1

        except Exception as e:
            evaluator.logger.error(f"Error processing batch result: {e}")
            failed_count += 1
            continue

    # Save processed evaluations
    evaluator._save_evaluations()
    evaluator.logger.info(f"Batch results processed and saved: {processed_count} successful, {failed_count} failed")


def cancel_batch(evaluator):
    """Cancel the current batch job."""
    if not evaluator.batch_id:
        raise ValueError("No batch ID found")

    evaluator.client.cancel_batch(evaluator.batch_id)
    evaluator.logger.info(f"Batch {evaluator.batch_id} cancelled")
    return evaluator
