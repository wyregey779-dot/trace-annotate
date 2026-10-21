from trace_annotate.embodied.motion import propose_idle_pauses, snap_boundary_to_motion


def test_propose_idle_pauses():
    # 0..2s moving, 2..6s idle, 6..10s moving
    series = [
        {"timestamp": 0.0, "is_moving": True, "energy": 10.0},
        {"timestamp": 1.0, "is_moving": True, "energy": 8.0},
        {"timestamp": 2.0, "is_moving": False, "energy": 1.0},
        {"timestamp": 3.0, "is_moving": False, "energy": 1.0},
        {"timestamp": 4.0, "is_moving": False, "energy": 0.5},
        {"timestamp": 5.0, "is_moving": False, "energy": 0.8},
        {"timestamp": 6.0, "is_moving": True, "energy": 7.0},
        {"timestamp": 7.0, "is_moving": True, "energy": 9.0},
    ]
    pauses = propose_idle_pauses(series, min_pause_s=2.0)
    assert len(pauses) == 1
    assert pauses[0]["start"] == 2.0
    assert pauses[0]["end"] == 6.0
    assert pauses[0]["duration"] == 4.0


def test_snap_boundary_to_motion():
    # Motion onset at t=2.0
    series = [
        {"timestamp": 1.0, "is_moving": False, "energy": 0.5},
        {"timestamp": 1.5, "is_moving": False, "energy": 0.5},
        {"timestamp": 2.0, "is_moving": True, "energy": 12.0},
        {"timestamp": 2.5, "is_moving": True, "energy": 14.0},
    ]
    # Rough click at 1.8s should snap to 2.0s
    snapped = snap_boundary_to_motion(series, target_ts=1.8, window_s=0.5, snap_to_onset=True)
    assert snapped == 2.0
