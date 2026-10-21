from trace_annotate.embodied.complement import compute_idle_complement


def test_idle_complement():
    working = [
        {"start": 10.0, "end": 20.0},
        {"start": 30.0, "end": 45.0}
    ]
    idle = compute_idle_complement(working, total_duration=60.0)
    assert len(idle) == 3
    assert idle[0] == {"start": 0.0, "end": 10.0}
    assert idle[1] == {"start": 20.0, "end": 30.0}
    assert idle[2] == {"start": 45.0, "end": 60.0}
