from trace_annotate.embodied.validator import validate_embodied_doc


def test_embodied_validator():
    # Valid doc
    valid = {
        "clip": "test.mp4",
        "clip_status": "正常",
        "working": [
            {
                "start": 1.0,
                "end": 5.0,
                "step_number": 1,
                "verdict": "对"
            },
            {
                "start": 6.0,
                "end": 10.0,
                "step_number": 2,
                "verdict": "错",
                "error_type": "用错物体",
                "evidence": "t=7s 用了改锥应为扳手"
            }
        ]
    }
    errs = validate_embodied_doc(valid)
    assert errs == []

    # Invalid doc: verdict 错 without evidence
    invalid = {
        "clip": "test_bad.mp4",
        "clip_status": "正常",
        "working": [
            {
                "start": 1.0,
                "end": 5.0,
                "step_number": 1,
                "verdict": "错",
                "error_type": "用错物体",
                "evidence": ""
            }
        ]
    }
    errs2 = validate_embodied_doc(invalid)
    assert len(errs2) > 0
    assert any("evidence" in e for e in errs2)
