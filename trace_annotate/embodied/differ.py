"""Reconcile observed chronological steps with an independently supplied procedure."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ._intervals import interval, overlap_errors

_MAX_EXPECTED_STEPS = 100_000  # Resource bound, not an annotation quality rule.


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass
class ProcedureSpec:
    name: str = "default_procedure"
    steps: List[int] = field(default_factory=list)
    step_names: Dict[int, str] = field(default_factory=dict)
    required_steps: List[int] = field(default_factory=list)
    allowed_transitions: Optional[Dict[int, List[int]]] = None

    def __post_init__(self):
        if not isinstance(self.steps, list) or not isinstance(self.step_names, dict) or not isinstance(self.required_steps, list):
            raise ValueError("procedure steps/required_steps must be arrays and step_names a mapping")
        if not self.steps and self.step_names:
            if any(not _positive_int(step) for step in self.step_names):
                raise ValueError("step_names keys must be positive integers")
            self.steps = sorted(self.step_names)
        if not self.required_steps and self.steps:
            self.required_steps = list(self.steps)
        if (not self.steps or len(self.steps) > _MAX_EXPECTED_STEPS
                or any(not _positive_int(step) for step in self.steps) or len(set(self.steps)) != len(self.steps)):
            raise ValueError("procedure steps must be distinct positive integers")
        if any(not _positive_int(step) or step not in self.steps for step in self.required_steps):
            raise ValueError("required_steps must belong to the procedure")
        if self.allowed_transitions is not None:
            if not isinstance(self.allowed_transitions, dict):
                raise ValueError("allowed_transitions must be a mapping")
            for step, successors in self.allowed_transitions.items():
                if (not _positive_int(step) or step not in self.steps or not isinstance(successors, list)
                        or any(not _positive_int(item) or item not in self.steps for item in successors)):
                    raise ValueError("allowed_transitions must refer to known procedure steps")


def diff_expected_steps(working_segments: List[Dict[str, Any]],
                        expected_step_count: Optional[int] = None,
                        procedure: Optional[ProcedureSpec] = None) -> Dict[str, Any]:
    """Use actual time order when available; never infer completeness from max(step).

    Step-only inputs retain a sequence diagnostic. A complete gate requires a
    separately supplied N or ProcedureSpec. Invalid time order is also a failure,
    even if the JSON array was sorted by the desired step numbers.
    """
    if not isinstance(working_segments, list):
        raise ValueError("working_segments must be an array")
    if expected_step_count is not None and not _positive_int(expected_step_count):
        raise ValueError("expected_step_count must be a positive integer")
    if expected_step_count is not None and expected_step_count > _MAX_EXPECTED_STEPS:
        raise ValueError("STEP_LIMIT: expected_step_count exceeds the 100000-step resource limit")
    if procedure is not None and not isinstance(procedure, ProcedureSpec):
        raise ValueError("procedure must be a ProcedureSpec")
    if procedure is not None and expected_step_count is not None:
        if procedure.steps != list(range(1, expected_step_count + 1)):
            raise ValueError("expected_step_count conflicts with procedure")
    known = procedure is not None or expected_step_count is not None
    required = procedure.required_steps if procedure else range(1, expected_step_count + 1) if known else []
    positions = {step: index for index, step in enumerate(procedure.steps)} if procedure else {}
    valid_set = set(procedure.steps) if procedure else None
    invalid: List[Dict[str, Any]] = []
    times = []
    temporal_errors: List[str] = []
    has_times = any(isinstance(segment, dict) and ("start" in segment or "end" in segment) for segment in working_segments)
    valid = {}
    for index, segment in enumerate(working_segments):
        step = segment.get("step_number") if isinstance(segment, dict) else None
        reason = None
        if not _positive_int(step):
            reason = "non_positive_integer_step"
        elif valid_set is not None and step not in valid_set:
            reason = "not_in_procedure"
        elif expected_step_count is not None and step > expected_step_count:
            reason = f"out_of_range (1..{expected_step_count})"
        if reason:
            invalid.append({"index": index, "step_number": step, "reason": reason})
        else:
            valid[index] = step
        if has_times:
            try:
                times.append((*interval(segment, f"working[{index}]"), index))
            except ValueError as exc:
                temporal_errors.append(str(exc))
    if has_times:
        temporal_errors.extend(overlap_errors(times, "working"))
        ordered = sorted(times)
        if [item[2] for item in ordered] != [item[2] for item in times]:
            temporal_errors.append("TIME_ORDER: working array is not in chronological order")
        indices = [item[2] for item in ordered]
    else:
        indices = list(range(len(working_segments)))
    sequence = [(index, valid[index]) for index in indices if index in valid]
    counts: Dict[int, int] = {}
    for _, step in valid.items():
        counts[step] = counts.get(step, 0) + 1
    missing = sorted(step for step in required if step not in counts)
    duplicate = sorted(step for step, count in counts.items() if count > 1)
    out_of_order = []
    for previous, current in zip(sequence, sequence[1:]):
        prev_index, prev_step = previous
        index, step = current
        if procedure and procedure.allowed_transitions is not None:
            bad = step not in procedure.allowed_transitions.get(prev_step, [])
        else:
            bad = positions[step] < positions[prev_step] if procedure else step < prev_step
        if bad:
            out_of_order.append({"index": index, "previous_index": prev_index,
                                 "previous_step": prev_step, "current_step": step})
    observed_valid = not (invalid or temporal_errors or out_of_order or duplicate) and bool(working_segments)
    return {"missing": missing, "out_of_order": out_of_order, "duplicate": duplicate,
            "invalid_steps": invalid, "temporal_errors": temporal_errors,
            "sequence": [step for _, step in sequence], "coverage_known": known,
            "observed_sequence_valid": observed_valid,
            "is_perfect_sequence": known and observed_valid and not missing}
