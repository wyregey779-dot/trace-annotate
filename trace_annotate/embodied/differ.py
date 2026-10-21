"""
Expected Step Differ & Reconciliation.
Automatically compares working segment step_number annotations against
expected sequential procedure S = [1, 2, ..., N] or a custom ProcedureSpec.
Detects missing steps, out-of-order inversions, duplicates, and out-of-range/invalid steps.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field


@dataclass
class ProcedureSpec:
    """
    Standard definition of an embodied or task procedure.
    """
    name: str = "default_procedure"
    steps: List[int] = field(default_factory=list)
    step_names: Dict[int, str] = field(default_factory=dict)
    required_steps: List[int] = field(default_factory=list)
    allowed_transitions: Optional[Dict[int, List[int]]] = None

    def __post_init__(self):
        if not self.steps and self.step_names:
            self.steps = sorted(self.step_names.keys())
        if not self.required_steps and self.steps:
            self.required_steps = list(self.steps)


def diff_expected_steps(
    working_segments: List[Dict[str, Any]],
    expected_step_count: Optional[int] = None,
    procedure: Optional[ProcedureSpec] = None
) -> Dict[str, Any]:
    """
    Diff annotated step numbers against expected workflow.
    
    Args:
        working_segments: list of working segment dicts with 'step_number'
        expected_step_count: total expected steps (1..N). If omitted, inferred from procedure or max step.
        procedure: optional ProcedureSpec defining valid steps, required steps, and allowed transitions.
        
    Returns:
        Dict containing:
          - 'missing': list of missing step numbers
          - 'out_of_order': list of sequence violations
          - 'duplicate': list of repeated step numbers
          - 'invalid_steps': list of steps outside the expected procedure or range
          - 'is_perfect_sequence': bool indicating if sequence strictly follows procedure without error
    """
    nums_with_indices = [
        (i, w.get("step_number"))
        for i, w in enumerate(working_segments)
    ]

    invalid_steps: List[Dict[str, Any]] = []
    valid_nums: List[int] = []

    if procedure is not None:
        valid_set = set(procedure.steps)
        required_set = set(procedure.required_steps)

        for idx, s in nums_with_indices:
            if not isinstance(s, int):
                invalid_steps.append({
                    "index": idx,
                    "step_number": s,
                    "reason": "non_integer_step"
                })
            elif s not in valid_set:
                invalid_steps.append({
                    "index": idx,
                    "step_number": s,
                    "reason": f"not_in_procedure (valid={sorted(list(valid_set))})"
                })
            else:
                valid_nums.append(s)

        counts: Dict[int, int] = {}
        for n in valid_nums:
            counts[n] = counts.get(n, 0) + 1

        duplicates = sorted([n for n, c in counts.items() if c > 1])
        missing = sorted([n for n in required_set if n not in counts])

        out_of_order: List[Dict[str, Any]] = []
        for i in range(1, len(valid_nums)):
            prev_s = valid_nums[i - 1]
            curr_s = valid_nums[i]

            if procedure.allowed_transitions is not None:
                allowed_next = procedure.allowed_transitions.get(prev_s, [])
                if curr_s not in allowed_next:
                    out_of_order.append({
                        "index": i,
                        "previous_step": prev_s,
                        "current_step": curr_s,
                        "reason": f"transition_{prev_s}_to_{curr_s}_not_allowed"
                    })
            else:
                if curr_s < prev_s:
                    out_of_order.append({
                        "index": i,
                        "previous_step": prev_s,
                        "current_step": curr_s
                    })

    elif expected_step_count is not None:
        valid_range = set(range(1, expected_step_count + 1))

        for idx, s in nums_with_indices:
            if not isinstance(s, int):
                invalid_steps.append({
                    "index": idx,
                    "step_number": s,
                    "reason": "non_integer_step"
                })
            elif s not in valid_range:
                invalid_steps.append({
                    "index": idx,
                    "step_number": s,
                    "reason": f"out_of_range (1..{expected_step_count})"
                })
            else:
                valid_nums.append(s)

        counts = {}
        for n in valid_nums:
            counts[n] = counts.get(n, 0) + 1

        duplicates = sorted([n for n, c in counts.items() if c > 1])
        missing = [n for n in range(1, expected_step_count + 1) if n not in counts]

        out_of_order = []
        for i in range(1, len(valid_nums)):
            if valid_nums[i] < valid_nums[i - 1]:
                out_of_order.append({
                    "index": i,
                    "previous_step": valid_nums[i - 1],
                    "current_step": valid_nums[i]
                })

    else:
        # No procedure or expected_step_count provided
        for idx, s in nums_with_indices:
            if isinstance(s, int):
                valid_nums.append(s)
            else:
                invalid_steps.append({
                    "index": idx,
                    "step_number": s,
                    "reason": "non_integer_step"
                })

        counts = {}
        for n in valid_nums:
            counts[n] = counts.get(n, 0) + 1

        duplicates = sorted([n for n, c in counts.items() if c > 1])
        n_steps = max(valid_nums) if valid_nums else 0
        missing = [n for n in range(1, n_steps + 1) if n not in counts]

        out_of_order = []
        for i in range(1, len(valid_nums)):
            if valid_nums[i] < valid_nums[i - 1]:
                out_of_order.append({
                    "index": i,
                    "previous_step": valid_nums[i - 1],
                    "current_step": valid_nums[i]
                })

    is_perfect = (
        len(missing) == 0
        and len(out_of_order) == 0
        and len(duplicates) == 0
        and len(invalid_steps) == 0
    )

    return {
        "missing": missing,
        "out_of_order": out_of_order,
        "duplicate": duplicates,
        "invalid_steps": invalid_steps,
        "is_perfect_sequence": is_perfect
    }
