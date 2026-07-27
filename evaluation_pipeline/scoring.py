# evaluation_pipeline/scoring.py

"""
Final score generation and evaluation status tracking.

Plain functions operating on an Evaluator instance (delegation pattern,
same as data.py). The Evaluator methods are one-line delegates to these.
"""

import json
from typing import Any, Dict

from . import utils


def generate_final_scores(evaluator):
    """Generate final scores using adjudication if available, otherwise average."""
    if not evaluator.evaluations:
        raise ValueError("No evaluations found. Cannot generate final scores.")

    final_scores = {}
    incomplete_sessions = [] # Sessions with < 2 evaluations
    processing_errors = [] # Sessions with processing errors

    for session_id, evals in evaluator.evaluations.items():
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
            evaluator.logger.error(f"Error processing final score for session {session_id}: {e}")
            continue

    # Save results
    evaluator.config.dirs.evaluation_results.mkdir(parents=True, exist_ok=True)

    output_path = evaluator.config.dirs.evaluation_results / f"{evaluator.config.run_id}_final_scores.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_scores, f, indent=2)

    if incomplete_sessions:
        incomplete_path = evaluator.config.dirs.evaluation_results / f"{evaluator.config.run_id}_incomplete_sessions.json"
        with open(incomplete_path, 'w', encoding='utf-8') as f:
            json.dump(incomplete_sessions, f, indent=2)
        evaluator.logger.warning(f"{len(incomplete_sessions)} session(s) incomplete - saved to {incomplete_path}")

    if processing_errors:
        errors_path = evaluator.config.dirs.evaluation_results / f"{evaluator.config.run_id}_processing_errors.json"
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(processing_errors, f, indent=2)
        evaluator.logger.warning(f"Processing errors saved to {errors_path}")

    evaluator.logger.info(f"Final scores saved to {output_path}")
    evaluator.logger.info(f"Total sessions scored: {len(final_scores)}")

    if incomplete_sessions:
        evaluator.logger.info(f"  {len(incomplete_sessions)} sessions excluded due to incomplete evaluations")

    return evaluator


def check_evaluation_status(evaluator) -> Dict[str, Any]:
    """Check the status of evaluations across all sessions."""
    if evaluator.session_data is None:
        evaluator.logger.warning("Session data not loaded. Cannot check evaluation status.")
        return {}

    # Initialize status dict
    status = {
        'total_sessions': len(evaluator.session_data),
        'not_started': [],
        'in_progress': [],
        'complete': [],
        'needs_adjudication': [],
        'ready_for_final_scores': [],
        'has_dynamic_prompts': len(evaluator.dynamic_prompts) > 0,
        'batch_file_created': evaluator.batch_file_path is not None and evaluator.batch_file_path.exists(),
        'batch_uploaded': evaluator.batch_id is not None
    }

    # Categorize each session
    for session_id in evaluator.session_data['session_id']:
        eval_count = len(evaluator.evaluations.get(session_id, []))

        if eval_count == 0:
            status['not_started'].append(session_id)
        elif eval_count == 1:
            status['in_progress'].append(session_id)
        elif eval_count >= 2:
            status['complete'].append(session_id)

            # Check if needs adjudication
            needs_adj, reason = utils.needs_adjudication(
                evaluator.evaluations[session_id][0][0],
                evaluator.evaluations[session_id][1][0]
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
        next_action = f"Run generate_final_scores() to complete the pipeline if you haven't already; check {evaluator.config.dirs.evaluation_results} for existing final scores"
    else:
        next_action = "No action needed"

    status['next_action'] = next_action

    return status
