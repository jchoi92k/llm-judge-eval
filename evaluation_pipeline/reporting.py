# evaluation_pipeline/reporting.py

"""
Diagnostics and reporting: pipeline status text, prompt sanity checks,
and actual cost reporting.

Plain functions operating on an Evaluator instance (delegation pattern,
same as data.py). The Evaluator methods are one-line delegates to these.
"""

import pickle
from typing import Any, Dict


def status_text(evaluator) -> str:
    """
    Build checklist-style status text for the evaluation pipeline (Evaluator.__repr__).
    ✓ = Completed; good to go
    △ = In progress or optional
    ✗ = Not started or missing; action needed
    """
    lines = [f"Evaluator(run_id={evaluator.config.run_id}, model={evaluator.config.model.model_name})"]
    lines.append("\nStatus:")

    # Check data loading
    if evaluator.session_data is not None:
        lines.append(f"  ✓ Session data loaded ({len(evaluator.session_data)} sessions)")
    else:
        lines.append("  ✗ Session data not loaded")

    # Check if human evaluation exists; optional
    if evaluator.human_evaluation is not None:
        lines.append(f"  ✓ Human evaluation loaded ({len(evaluator.human_evaluation)} samples) [optional]")
    else:
        lines.append("  △ Human evaluation not loaded [optional]")

    # Check RAG data load
    if evaluator.rag_dictionary is not None and evaluator.rag_embeddings is not None:
        lines.append(f"  ✓ RAG data loaded ({len(evaluator.rag_dictionary)} entries)")
    else:
        lines.append("  ✗ RAG data not loaded")

    # Check if guidelines have already been generated
    if "guideline_final.txt" in evaluator.guidelines:
        lines.append("  ✓ Guidelines available (shared across runs)")
    else:
        lines.append("  ✗ Guidelines not generated")

    # Check evaluations using check_evaluation_status
    if evaluator.session_data is not None:
        status = evaluator.check_evaluation_status()

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

    if evaluator.session_data is not None:
        status = evaluator.check_evaluation_status()

        # Check if guidelines exist
        guidelines_exist = "guideline_final.txt" in evaluator.guidelines

        # If guidelines exist, show evaluation next steps
        if guidelines_exist:
            lines.append(f"  → {status['next_action']}")

        # If no guidelines, check human eval status
        elif evaluator.human_evaluation is None:
            lines.append("  → Provide human evaluation data via config.toml (optional but recommended)")
            lines.append("  → Or proceed to run generate_evaluation_guidelines()")
        else:
            lines.append("  → Run generate_evaluation_guidelines()")

    return "\n".join(lines)


def sanity_check_prompts(evaluator, n_examples: int = 3):
    """
    Sanity check generated prompts for correct few-shot insertion and image handling.

    Checks:
    1. Few-shot examples: Whether the "Example Evaluations from Human Raters" section
       contains actual content (not empty).
    2. Images: Whether sessions with image data have input_image content parts in the prompt.
    """
    if not evaluator.dynamic_prompts:
        print("No dynamic prompts generated. Run generate_dynamic_prompts() first.")
        return

    print(f"{'='*70}")
    print(f"PROMPT SANITY CHECK — {len(evaluator.dynamic_prompts)} prompts")
    print(f"{'='*70}\n")

    # ---- Check 1: Few-shot examples ----
    print("## 1. Few-Shot Human Examples\n")

    few_shot_marker = "### Example Evaluations from Human Raters"
    next_section_marker = "### Example Tutor Conversations"

    has_examples = 0
    empty_examples = 0
    example_char_counts = []

    for session_id, prompt in evaluator.dynamic_prompts.items():
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

    print(f"  Sessions with few-shot content:  {has_examples}/{len(evaluator.dynamic_prompts)}")
    print(f"  Sessions without few-shot content: {empty_examples}/{len(evaluator.dynamic_prompts)}")

    if has_examples == 0:
        print("\n  ⚠ WARNING: No prompts contain few-shot examples!")
    elif empty_examples > 0:
        print(f"\n  ⚠ WARNING: {empty_examples} prompts are missing few-shot examples")

    # Show examples
    sorted_counts = sorted(example_char_counts, key=lambda x: x[1], reverse=True)
    print(f"\n  Top {min(n_examples, len(sorted_counts))} by few-shot section size:")
    for sid, count in sorted_counts[:n_examples]:
        print(f"    {str(sid)[:20]}...  {count:,} chars")

    if empty_examples > 0:
        print("\n  Empty few-shot examples:")
        empty = [x for x in sorted_counts if x[1] <= 50]
        for sid, count in empty[:n_examples]:
            print(f"    {str(sid)[:20]}...  {count} chars")

    # ---- Check 2: Image handling ----
    print("\n## 2. Image Handling\n")

    sessions_with_images = 0
    prompts_with_images = 0
    image_mismatches = []
    image_examples = []

    for _, row in evaluator.session_data.iterrows():
        session_id = row["session_id"]
        if session_id not in evaluator.dynamic_prompts:
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
        prompt = evaluator.dynamic_prompts[session_id]
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

    print(f"  Sessions with image data:    {sessions_with_images}/{len(evaluator.dynamic_prompts)}")
    print(f"  Prompts with image parts:    {prompts_with_images}/{len(evaluator.dynamic_prompts)}")

    if image_mismatches:
        print(f"\n  ⚠ MISMATCHES FOUND: {len(image_mismatches)}")
        for sid, n_data, n_prompt, reason in image_mismatches[:n_examples]:
            print(f"    {str(sid)[:20]}...  data: {n_data} images, prompt: {n_prompt} images — {reason}")
    else:
        if sessions_with_images > 0:
            print("\n  ✓ All image counts match between data and prompts")
        else:
            print("\n  (No sessions with images in current prompt set)")

    if image_examples:
        print(f"\n  Sample sessions with images (up to {n_examples}):")
        for sid, n_data, n_prompt in image_examples[:n_examples]:
            prompt = evaluator.dynamic_prompts[sid]
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


def failures_summary(evaluator) -> list:
    """
    Print a summary of evaluations skipped during this session's runs
    (parse failures, batch retrieval issues) and return the raw entries.
    """
    failures = evaluator.failures

    if not failures:
        print("No recorded failures — every evaluation call this session was parsed and stored.")
        return failures

    print(f"{'='*70}")
    print(f"SKIPPED EVALUATIONS — {len(failures)} total (in-memory, this session)")
    print(f"{'='*70}\n")

    # Group by reason
    by_reason = {}
    for f in failures:
        by_reason.setdefault(f["reason"], []).append(f)

    for reason, entries in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        print(f"  {reason}: {len(entries)}")
        for f in entries:
            sid = f["session_id"]
            sid_display = f"{str(sid)[:24]}..." if sid is not None and len(str(sid)) > 24 else (sid if sid is not None else "(unknown session)")
            detail = str(f["detail"])
            detail_display = f"{detail[:60]}..." if len(detail) > 60 else detail
            print(f"    [{f['stage']}] {sid_display}: {detail_display}")
        print()

    # Sessions affected (excluding unknowns)
    affected = sorted({str(f["session_id"]) for f in failures if f["session_id"] is not None})
    if affected:
        print(f"  Affected sessions ({len(affected)}): re-run flex_evaluate() or the batch flow")
        print("  to fill missing evaluations; check_evaluation_status() shows what's incomplete.")

    return failures


def report_actual_cost(evaluator, all_runs: bool = False) -> Dict[str, Any]:
    """Report actual token usage and cost from completed evaluations."""
    results = []

    if all_runs:
        pkl_files = sorted(evaluator.config.dirs.evaluation_results.glob("*_evaluations.pkl"))
    else:
        pkl_files = [evaluator.config.dirs.evaluation_results / f"{evaluator.config.run_id}_evaluations.pkl"]
        pkl_files = [p for p in pkl_files if p.exists()]

    if not pkl_files and evaluator.evaluations:
        # Use in-memory evaluations
        pkl_files = []
        evals_to_scan = [("current", evaluator.evaluations)]
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
            uncached * evaluator.config.model.input_token_price
            + total_cached * evaluator.config.model.cached_token_price
            + total_output * evaluator.config.model.output_token_price
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
            uncached_grand * evaluator.config.model.input_token_price
            + grand_cached * evaluator.config.model.cached_token_price
            + grand_output * evaluator.config.model.output_token_price
        )
        print("=" * 78)
        print(f"{'TOTAL':<20} {'':>8} {grand_evals:>6} {grand_input:>12,} {grand_output:>12,} {'$' + f'{grand_cost:.2f}':>10}")
        print(f"\nCached input tokens: {grand_cached:,} ({grand_cached/grand_input*100:.0f}%)" if grand_input else "")

    return results
