from trace_annotate.embodied.differ import diff_expected_steps


def test_step_diff():
    # Sequence with missing step 2 and inversion (3 before 2)
    working = [
        {"step_number": 1},
        {"step_number": 3},
        {"step_number": 2}
    ]
    diff = diff_expected_steps(working, expected_step_count=3)
    assert diff["missing"] == []
    assert len(diff["out_of_order"]) == 1
    assert diff["out_of_order"][0]["previous_step"] == 3
    assert diff["out_of_order"][0]["current_step"] == 2
    assert diff["is_perfect_sequence"] is False

    # Perfect sequence
    perfect = [{"step_number": 1}, {"step_number": 2}, {"step_number": 3}]
    diff_p = diff_expected_steps(perfect, expected_step_count=3)
    assert diff_p["is_perfect_sequence"] is True
