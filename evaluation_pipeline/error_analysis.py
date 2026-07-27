# evaluation_pipeline/error_analysis.py

"""
Sniff-test error analysis utilities.

Compares LLM evaluation scores against human review scores on a small
human-reviewed subset, computes agreement metrics, and builds prompts
for LLM-assisted error analysis to refine evaluation guidelines.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# COLUMN MAPPING
# ============================================================================

# Default mapping from common human-review column names to rubric-format names.
# Users can override by passing a custom mapping dict.
DEFAULT_HUMAN_COLUMN_MAP = {
    # Mathematical Accuracy
    "Accuracy - Validity": "Mathematical_Accuracy_Validity",
    "Accuracy - Clarity and Labeling": "Mathematical_Accuracy_Clarity_and_Labeling",
    "Accuracy- Justification and Explanation": "Mathematical_Accuracy_Justification_and_Explanation",
    "Accuracy- Justification and Explanation ": "Mathematical_Accuracy_Justification_and_Explanation",
    # Pedagogical Quality
    "PQ- Problem-Solving Strategies": "Pedagogical_Quality_Problem_Solving_Strategies",
    "PQ - Relevance": "Pedagogical_Quality_Relevance",
    "PQ- Scaffolded Support": "Pedagogical_Quality_Scaffolded_Support",
    "PQ- Clarity of Explanation": "Pedagogical_Quality_Clarity_of_Explanation",
    "PQ- Feedback": "Pedagogical_Quality_Feedback",
    "PQ- Motivational Engagement": "Pedagogical_Quality_Motivational_Engagement",
    # Equity and Fairness
    "EF- Language Neutrality": "Equity_and_Fairness_Language_neutrality",
    "EF - Feedback Tone": "Equity_and_Fairness_Feedback_tone",
    "EF- Cultural Relevance": "Equity_and_Fairness_Cultural_relevance",
}

# Rubric criteria organized by domain for structured comparison
RUBRIC_STRUCTURE = {
    "Mathematical_Accuracy": [
        "Validity",
        "Clarity_and_Labeling",
        "Justification_and_Explanation",
    ],
    "Pedagogical_Quality": [
        "Problem_Solving_Strategies",
        "Relevance",
        "Scaffolded_Support",
        "Clarity_of_Explanation",
        "Feedback",
        "Motivational_Engagement",
    ],
    "Equity_and_Fairness": [
        "Language_neutrality",
        "Feedback_tone",
        "Cultural_relevance",
    ],
}


# ============================================================================
# SCORE COMPARISON
# ============================================================================

def compare_scores(
    llm_scores: Dict[str, Dict],
    human_scores_df: pd.DataFrame,
    session_id_col: str = "session_id",
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Create a comparison table of LLM vs human scores per session per criterion.

    Args:
        llm_scores: Dict mapping session_id -> evaluation result dict
                    (with 'scores' key containing nested domain/criterion scores)
        human_scores_df: DataFrame with human scores. Must have a session ID column
                         and score columns (either already in rubric format or
                         mappable via column_map).
        session_id_col: Name of the session ID column in human_scores_df.
        column_map: Mapping from human CSV column names to rubric-format names.
                    Defaults to DEFAULT_HUMAN_COLUMN_MAP.

    Returns:
        DataFrame with columns: session_id, domain, criterion, human_score,
        llm_score, difference (llm - human), abs_difference
    """
    if column_map is None:
        column_map = DEFAULT_HUMAN_COLUMN_MAP

    rows = []

    for _, human_row in human_scores_df.iterrows():
        sid = str(human_row[session_id_col])

        if sid not in llm_scores:
            logger.warning(f"Session {sid} not found in LLM scores, skipping")
            continue

        llm_result = llm_scores[sid]
        llm_score_dict = llm_result.get("scores", {})

        for domain, criteria in RUBRIC_STRUCTURE.items():
            for criterion in criteria:
                rubric_col = f"{domain}_{criterion}"

                # Get human score — try rubric format first, then mapped
                human_val = None
                if rubric_col in human_row.index:
                    human_val = human_row[rubric_col]
                else:
                    # Try reverse lookup in column_map
                    for orig_col, mapped_col in column_map.items():
                        if mapped_col == rubric_col and orig_col in human_row.index:
                            human_val = human_row[orig_col]
                            break

                if human_val is None or (isinstance(human_val, float) and np.isnan(human_val)):
                    continue

                # Get LLM score
                llm_val = llm_score_dict.get(domain, {}).get(criterion)

                if llm_val is None:
                    continue

                human_val = int(human_val)
                llm_val_num = float(llm_val) if not isinstance(llm_val, (int, float)) else llm_val

                rows.append({
                    "session_id": sid,
                    "domain": domain,
                    "criterion": criterion,
                    "human_score": human_val,
                    "llm_score": llm_val_num,
                    "difference": llm_val_num - human_val,
                    "abs_difference": abs(llm_val_num - human_val),
                })

    return pd.DataFrame(rows)


# ============================================================================
# AGREEMENT METRICS
# ============================================================================

def compute_agreement_metrics(comparison_df: pd.DataFrame) -> Dict:
    """
    Compute agreement statistics from a comparison DataFrame.

    Returns dict with:
      - overall: exact_match_pct, mean_abs_diff, mean_diff (bias direction)
      - by_criterion: per-criterion stats
      - by_domain: per-domain stats
      - worst_criteria: criteria sorted by mean absolute difference (descending)
    """
    if comparison_df.empty:
        return {"overall": {}, "by_criterion": {}, "by_domain": {}, "worst_criteria": []}

    # Overall
    exact_matches = (comparison_df["abs_difference"] == 0).sum()
    within_one = (comparison_df["abs_difference"] <= 1).sum()
    total = len(comparison_df)

    overall = {
        "n_comparisons": total,
        "exact_match_pct": round(exact_matches / total * 100, 1),
        "within_one_pct": round(within_one / total * 100, 1),
        "mean_abs_difference": round(comparison_df["abs_difference"].mean(), 3),
        "mean_difference": round(comparison_df["difference"].mean(), 3),
        "bias_direction": (
            "LLM scores higher" if comparison_df["difference"].mean() > 0.05
            else "LLM scores lower" if comparison_df["difference"].mean() < -0.05
            else "No clear bias"
        ),
    }

    # By criterion (domain_criterion key)
    by_criterion = {}
    for (domain, criterion), group in comparison_df.groupby(["domain", "criterion"]):
        key = f"{domain}_{criterion}"
        n = len(group)
        by_criterion[key] = {
            "n": n,
            "exact_match_pct": round((group["abs_difference"] == 0).sum() / n * 100, 1),
            "mean_abs_difference": round(group["abs_difference"].mean(), 3),
            "mean_difference": round(group["difference"].mean(), 3),
            "bias_direction": (
                "LLM higher" if group["difference"].mean() > 0.05
                else "LLM lower" if group["difference"].mean() < -0.05
                else "Aligned"
            ),
        }

    # By domain
    by_domain = {}
    for domain, group in comparison_df.groupby("domain"):
        n = len(group)
        by_domain[domain] = {
            "n": n,
            "exact_match_pct": round((group["abs_difference"] == 0).sum() / n * 100, 1),
            "mean_abs_difference": round(group["abs_difference"].mean(), 3),
            "mean_difference": round(group["difference"].mean(), 3),
        }

    # Worst criteria (sorted by mean absolute difference)
    worst = sorted(by_criterion.items(), key=lambda x: x[1]["mean_abs_difference"], reverse=True)
    worst_criteria = [{"criterion": k, **v} for k, v in worst]

    return {
        "overall": overall,
        "by_criterion": by_criterion,
        "by_domain": by_domain,
        "worst_criteria": worst_criteria,
    }


# ============================================================================
# REPORT FORMATTING
# ============================================================================

def format_comparison_table(comparison_df: pd.DataFrame) -> str:
    """Format comparison DataFrame as a readable markdown table."""
    if comparison_df.empty:
        return "No comparisons available."

    lines = ["| Session | Domain | Criterion | Human | LLM | Diff |",
             "|---------|--------|-----------|-------|-----|------|"]

    for _, row in comparison_df.iterrows():
        sid_short = str(row["session_id"])[:12]
        diff_str = f"{row['difference']:+.1f}" if row['difference'] != 0 else "0"
        lines.append(
            f"| {sid_short}... | {row['domain']} | {row['criterion']} | "
            f"{row['human_score']} | {row['llm_score']:.1f} | {diff_str} |"
        )

    return "\n".join(lines)


def format_metrics_summary(metrics: Dict) -> str:
    """Format agreement metrics as readable text."""
    overall = metrics.get("overall", {})
    if not overall:
        return "No metrics available."

    lines = [
        "## Overall Agreement",
        f"- Comparisons: {overall.get('n_comparisons', 0)}",
        f"- Exact match: {overall.get('exact_match_pct', 0)}%",
        f"- Within 1 point: {overall.get('within_one_pct', 0)}%",
        f"- Mean absolute difference: {overall.get('mean_abs_difference', 0)}",
        f"- Mean difference (bias): {overall.get('mean_difference', 0)} ({overall.get('bias_direction', 'N/A')})",
        "",
        "## By Domain",
    ]

    for domain, stats in metrics.get("by_domain", {}).items():
        lines.append(f"- **{domain}**: exact match {stats['exact_match_pct']}%, "
                      f"MAD {stats['mean_abs_difference']}, bias {stats['mean_difference']:+.3f}")

    lines.append("")
    lines.append("## Criteria Ranked by Disagreement (worst first)")

    for item in metrics.get("worst_criteria", [])[:6]:
        lines.append(f"- **{item['criterion']}**: MAD {item['mean_abs_difference']}, "
                      f"bias {item['mean_difference']:+.3f} ({item['bias_direction']}), "
                      f"exact match {item['exact_match_pct']}%")

    return "\n".join(lines)


def format_error_analysis_report(
    comparison_df: pd.DataFrame,
    metrics: Dict,
    llm_analysis: str,
) -> str:
    """
    Format a complete human-readable error analysis report.

    Args:
        comparison_df: Score comparison DataFrame
        metrics: Agreement metrics dict
        llm_analysis: LLM-generated analysis text

    Returns:
        Complete report as markdown string
    """
    sections = [
        "# Sniff Test — Error Analysis Report",
        "",
        format_metrics_summary(metrics),
        "",
        "## Detailed Comparison",
        format_comparison_table(comparison_df),
        "",
        "## LLM Error Analysis & Guideline Recommendations",
        llm_analysis,
    ]

    return "\n\n".join(sections)
