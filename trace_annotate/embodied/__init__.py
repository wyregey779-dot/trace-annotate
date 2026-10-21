"""
Embodied AI Video Annotation Suite.
Specialized for first-person / egocentric task video temporal segmentation,
working/idle phase division, expected step diffing, 7-class error diagnosis,
and inter-annotator temporal Cohen's Kappa agreement calculation.
"""
from trace_annotate.embodied.schema import (
    ClipStatus,
    ErrorType,
    Verdict,
    WorkingSegment,
    MidSegment,
    IdleSegment,
    EmbodiedAnnotationDoc,
)
from trace_annotate.embodied.complement import compute_idle_complement
from trace_annotate.embodied.differ import diff_expected_steps
from trace_annotate.embodied.agreement import calculate_temporal_kappa
from trace_annotate.embodied.validator import validate_embodied_doc, validate_embodied_export
from trace_annotate.embodied.label_studio import generate_label_studio_xml, parse_label_studio_export

__all__ = [
    "ClipStatus",
    "ErrorType",
    "Verdict",
    "WorkingSegment",
    "MidSegment",
    "IdleSegment",
    "EmbodiedAnnotationDoc",
    "compute_idle_complement",
    "diff_expected_steps",
    "calculate_temporal_kappa",
    "validate_embodied_doc",
    "validate_embodied_export",
    "generate_label_studio_xml",
    "parse_label_studio_export",
]
