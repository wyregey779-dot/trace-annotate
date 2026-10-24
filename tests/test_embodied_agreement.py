from trace_annotate.embodied.agreement import calculate_temporal_kappa


def test_temporal_kappa():
    # Identical annotators with both working and idle intervals -> Kappa = 1.0
    a = {"clip1": {"working": [{"start": 2.0, "end": 8.0}, {"start": 12.0, "end": 16.0}]}}
    b = {"clip1": {"working": [{"start": 2.0, "end": 8.0}, {"start": 12.0, "end": 16.0}]}}
    res = calculate_temporal_kappa(a, b, sample_rate_hz=2.0)
    assert res["kappa"] == 1.0
    # Time-only records support diagnostics, not full production acceptance.
    assert res["temporal_meets_trial_bar"] is True
    assert res["meets_trial_bar"] is False
    assert res["semantic_complete"] is False
    assert res["validation_errors"]

    # High agreement (slight boundary differences) -> Kappa >= 0.80
    c = {"clip1": {"working": [{"start": 2.5, "end": 8.0}, {"start": 11.5, "end": 16.0}]}}
    res2 = calculate_temporal_kappa(a, c, sample_rate_hz=2.0)
    assert res2["kappa"] >= 0.80
    assert res2["temporal_meets_trial_bar"] is True
    assert res2["meets_trial_bar"] is False
