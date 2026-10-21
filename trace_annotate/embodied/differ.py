"""
Expected Step Differ & Reconciliation.
Automatically compares working segment step_number annotations against
expected sequential procedure S = [1, 2, ..., N].
Detects missing steps, out-of-order inversions, and duplicates.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional


def diff_expected_steps(
    working_segments: List[Dict[str, Any]],
    expected_step_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Diff annotated step numbers against expected workflow.
    
    Args:
        working_segments: list of working segment dicts with 'step_number'
        expected_step_count: total expected steps (1..N). If omitted, inferred from max step.
        
    Returns:
        Dict containing:
          - 'missing': list of missing step numbers
          - 'out_of_order': list of sequence violations
          - 'duplicate': list of repeated step numbers
    """
    nums = [w.get("step_number") for w in working_segments if isinstance(w.get("step_number"), int)]

    counts: Dict[int, int] = {}
    for n in nums:
        counts[n] = counts.get(n, 0) + 1

    duplicates = sorted([n for n, c in counts.items() if c > 1])

    n_steps = expected_step_count or (max(nums) if nums else 0)
    missing = [n for n in range(1, n_steps + 1) if n not in counts]

    out_of_order = []
    for i in range(1, len(nums)):
        if nums[i] < nums[i - 1]:
            out_of_order.append({
                "index": i,
                "previous_step": nums[i - 1],
                "current_step": nums[i]
            })

    return {
        "missing": missing,
        "out_of_order": out_of_order,
        "duplicate": duplicates,
        "is_perfect_sequence": len(missing) == 0 and len(out_of_order) == 0 and len(duplicates) == 0
    }
