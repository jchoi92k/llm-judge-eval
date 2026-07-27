# evaluation_pipeline/calibration.py

"""
Sniff test calibration: evaluate the human-reviewed subset, compare with
human scores, and refine guidelines from the error analysis.

Plain functions operating on an Evaluator instance (delegation pattern,
same as data.py). The Evaluator methods are one-line delegates to these.
"""

import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .prompts import retrieve_similar
from . import utils


def generate_sniff_test_prompts(evaluator, sniff_data: pd.DataFrame):
    """
    Generate evaluation prompts for sniff test sessions with per-session
    few-shot exclusion: each session's prompt includes human-reviewed
    examples from OTHER sessions, never its own scores.

    The few-shot pool is evaluator.human_evaluation. For each session being
    evaluated, we exclude that session from the pool and sample from the
    remainder (up to n_human_rating_samples from config).
    """
    if evaluator.rag_dictionary is None or evaluator.rag_embeddings is None:
        raise ValueError("RAG data not loaded.")

    if not evaluator.guidelines.get("guideline_final.txt"):
        raise ValueError("Guidelines not generated.")

    final_guideline = evaluator.guidelines["guideline_final.txt"]
    n_few_shot = evaluator.config.evaluation_settings.n_human_rating_samples

    evaluator.dynamic_prompts = {}

    for _, row in tqdm(sniff_data.iterrows(), total=len(sniff_data), desc="Generating sniff test prompts"):
        session_id = str(row["session_id"])

        # --- Few-shot pool: exclude current session ---
        if evaluator.human_evaluation is not None and len(evaluator.human_evaluation) > 0:
            pool = evaluator.human_evaluation[
                evaluator.human_evaluation["session_id"].astype(str) != session_id
            ]
            # Sample up to n_few_shot from the remaining pool
            n_sample = min(n_few_shot, len(pool))
            if n_sample > 0:
                few_shot_samples = pool.sample(n=n_sample, random_state=evaluator.config.evaluation_settings.random_state)
            else:
                few_shot_samples = None
        else:
            few_shot_samples = None

        # --- RAG context ---
        row_string = utils.format_any_tabular_data(
            row[~row.index.isin(["image_data_base64"])],
            f"{evaluator.config.tool_settings.tool_name} Data",
        )
        row_string = utils.truncate_text_to_tokens(row_string, max_tokens=8000)
        embedding = evaluator.client.create_embedding(row_string)
        similar_entries = retrieve_similar(embedding, evaluator.rag_embeddings, top_k=1)
        similar_cid = similar_entries[0][0]
        rag_context = evaluator.rag_dictionary.get(similar_cid, "No similar content found.")

        # --- Build prompt with few-shot examples ---
        prompt = evaluator.prompt_builder.build_evaluation_prompt(
            row=row,
            rag_context=rag_context,
            guideline=final_guideline,
            human_evaluation_samples=few_shot_samples,
        )

        evaluator.dynamic_prompts[session_id] = prompt

    evaluator.logger.info(f"Generated {len(evaluator.dynamic_prompts)} sniff test prompts (with per-session few-shot exclusion)")


def run_sniff_test(
    evaluator,
    service_tier: str = "flex",
    auto_approve: bool = False,
    human_id_col: str = "session_id",
    column_map: dict = None,
) -> dict:
    """Run evaluation on the human-reviewed subset and compare with human scores."""
    from . import error_analysis
    from jinja2 import Template

    if evaluator.human_evaluation is None:
        raise ValueError(
            "Human evaluation data required for sniff test. "
            "Ensure human_evaluation path is set in config and the file exists."
        )

    if "guideline_final.txt" not in evaluator.guidelines:
        raise ValueError(
            "Guidelines must be generated before running sniff test. "
            "Call generate_evaluation_guidelines() first."
        )

    # Load the FULL human evaluation CSV (unsampled) for sniff test comparison.
    # evaluator.human_evaluation may be sampled via n_human_rating_samples (used for
    # guideline generation few-shot), but the sniff test needs all human reviews.
    full_human_eval = pd.read_csv(evaluator.config.file_paths.human_evaluation)
    if 'image_data_base64' in full_human_eval.columns:
        full_human_eval = full_human_eval.drop(columns=['image_data_base64'])

    # --- Step 1: Load ALL human-reviewed sessions from the full CSV ---
    # We bypass evaluator.session_data (which is sampled via n_samples) and load
    # the complete session CSV so the sniff test covers all human-reviewed sessions.
    full_session_data = pd.read_csv(evaluator.config.file_paths.session_data)
    if evaluator.data_prep_function:
        full_session_data = evaluator.data_prep_function(full_session_data, evaluator.config)
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

    evaluator.logger.info(
        f"Sniff test: {len(sniff_data)} sessions with human review data "
        f"(out of {len(human_session_ids)} human-reviewed)"
    )

    # --- Step 2: Evaluate with per-session few-shot exclusion ---
    original_session_data = evaluator.session_data
    original_evaluations = evaluator.evaluations.copy()
    original_dynamic_prompts = evaluator.dynamic_prompts.copy()

    evaluator.session_data = sniff_data.reset_index(drop=True)

    sniff_session_ids = set(sniff_data["session_id"].astype(str))
    for sid in sniff_session_ids:
        if sid in evaluator.evaluations:
            del evaluator.evaluations[sid]

    evaluator.dynamic_prompts = {}

    try:
        # Use sniff-test-specific prompt generation (with few-shot exclusion)
        generate_sniff_test_prompts(evaluator, evaluator.session_data)
        evaluator.flex_evaluate(
            service_tier=service_tier,
            adjudication=False,
            n_runs=2,
            auto_approve=auto_approve,
        )

        evaluator.generate_final_scores()

        scores_path = evaluator.config.dirs.evaluation_results / f"{evaluator.config.run_id}_final_scores.json"
        with open(scores_path, "r", encoding="utf-8") as f:
            llm_scores = json.load(f)

    finally:
        evaluator.session_data = original_session_data
        for sid in sniff_session_ids:
            if sid in evaluator.evaluations:
                original_evaluations[sid] = evaluator.evaluations[sid]
        evaluator.evaluations = original_evaluations
        evaluator.dynamic_prompts = original_dynamic_prompts

    # --- Step 3: Compare scores ---
    comparison_df = error_analysis.compare_scores(
        llm_scores=llm_scores,
        human_scores_df=full_human_eval,
        session_id_col=human_id_col,
        column_map=column_map,
    )

    if comparison_df.empty:
        evaluator.logger.warning("No score comparisons could be made. Check ID matching.")
        return {"comparison_df": comparison_df, "metrics": {}, "llm_analysis": "", "report": ""}

    metrics = error_analysis.compute_agreement_metrics(comparison_df)

    evaluator.logger.info(
        f"Sniff test comparison: {metrics['overall'].get('n_comparisons', 0)} score pairs, "
        f"{metrics['overall'].get('exact_match_pct', 0)}% exact match, "
        f"bias: {metrics['overall'].get('bias_direction', 'N/A')}"
    )

    # --- Step 4: LLM error analysis ---
    error_analysis_template_path = Path("inputs/prompts/error_analysis.j2")
    if not error_analysis_template_path.exists():
        evaluator.logger.warning("error_analysis.j2 template not found, skipping LLM analysis")
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
            guidelines=evaluator.guidelines.get("guideline_final.txt", ""),
            rubric=evaluator.prompt_builder.rubric_json,
            tool_description=evaluator.prompt_builder.tool_description,
            metrics_summary=error_analysis.format_metrics_summary(metrics),
            comparison_table=error_analysis.format_comparison_table(comparison_df),
            llm_explanations=llm_explanations if llm_explanations else None,
        )

        evaluator.logger.info("Running LLM error analysis...")
        response = evaluator.client.call(prompt, service_tier=service_tier)
        llm_analysis = response.output_text.strip()

    # --- Step 5: Generate and save report ---
    report = error_analysis.format_error_analysis_report(
        comparison_df, metrics, llm_analysis
    )

    report_path = (
        evaluator.config.dirs.evaluation_results
        / f"{evaluator.config.run_id}_sniff_test_report.md"
    )
    report_path.write_text(report, encoding="utf-8")
    evaluator.logger.info(f"Sniff test report saved to {report_path}")

    return {
        "comparison_df": comparison_df,
        "metrics": metrics,
        "llm_analysis": llm_analysis,
        "report": report,
    }


def refine_guidelines(
    evaluator,
    sniff_test_report: str,
    service_tier: str = "flex",
    guideline_name: str = "guideline_refined.txt",
) -> str:
    """Use the sniff test error analysis report to refine the evaluation guidelines."""
    current_guidelines = evaluator.guidelines.get("guideline_final.txt", "")
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

    evaluator.logger.info("Refining guidelines based on sniff test report...")
    response = evaluator.client.call(prompt, service_tier=service_tier)
    refined = response.output_text.strip()

    # Save refined guidelines
    refined_path = evaluator.config.dirs.evaluation_guidelines / guideline_name
    refined_path.write_text(refined, encoding="utf-8")
    evaluator.guidelines[guideline_name] = refined
    evaluator.logger.info(f"Refined guidelines saved to {refined_path}")

    return refined


def use_refined_guidelines(evaluator, guideline_name: str = "guideline_refined.txt"):
    """Swap the active final guideline to the refined version."""
    if guideline_name not in evaluator.guidelines:
        raise ValueError(
            f"Guideline '{guideline_name}' not found. "
            f"Available: {list(evaluator.guidelines.keys())}"
        )

    refined_text = evaluator.guidelines[guideline_name]

    # Back up current final guideline
    backup_name = "guideline_final_pre_refinement.txt"
    if "guideline_final.txt" in evaluator.guidelines and backup_name not in evaluator.guidelines:
        backup_path = evaluator.config.dirs.evaluation_guidelines / backup_name
        backup_path.write_text(evaluator.guidelines["guideline_final.txt"], encoding="utf-8")
        evaluator.guidelines[backup_name] = evaluator.guidelines["guideline_final.txt"]
        evaluator.logger.info(f"Backed up original guidelines to {backup_name}")

    # Overwrite final guideline
    evaluator.guidelines["guideline_final.txt"] = refined_text
    final_path = evaluator.config.dirs.evaluation_guidelines / "guideline_final.txt"
    final_path.write_text(refined_text, encoding="utf-8")
    evaluator.logger.info(f"Active guidelines updated from {guideline_name}")
