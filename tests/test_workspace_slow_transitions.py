"""Bounded long dissolve proposals retain actual frames and reject illumination."""
import itertools

import numpy as np
import pytest

from trace_annotate.workspace import transition_discovery as TD
from trace_annotate.workspace.errors import WorkspaceError


def textures():
    y, x = np.indices((48, 64))
    first = (((x // 4 + y // 6) % 2) * 180 + 35).astype(float)
    last = (((x // 9 + y // 3) % 2) * 150 + 70).astype(float)
    return first, last


def samples(images, *, offset=0):
    for position, image in enumerate(images):
        index = offset + position
        yield {"frame_index": str(index), "source_pts": str(1000 + index * index),
               "proxy_pts": str(index * 1001), "seconds": {"n": str(index * 1001), "d": "30000"},
               "frame": np.round(image).astype(np.uint8)}


def crossfade(count, *, nonlinear=False, inverted=False, holds=24):
    first, last = textures()
    if inverted:
        last = 255 - first
    weights = np.linspace(0, 1, count)
    if nonlinear:
        weights = weights ** 2
    return itertools.chain([first] * holds, ((1 - a) * first + a * last for a in weights),
                           [last] * holds)


def long_events(result):
    return [event for event in result["events"]
            if event["metrics"].get("measurement_method") in
            {"full_long_run_mixture", "split_long_run_mixture"}]


@pytest.mark.parametrize("count", [400, 1000, 2000])
def test_quantized_long_crossfade_retains_full_measured_interval(count):
    result = TD.discover_events(samples(crossfade(count), offset=300))
    event, = long_events(result)
    assert event["kind"] == "dissolve"
    assert int(event["interval"]["end_frame"]) - int(event["interval"]["start_frame"]) >= count * .95
    assert event["metrics"]["fit_interval"] == {
        "start_frame": event["interval"]["start_frame"], "end_frame": event["interval"]["end_frame"]}
    assert event["interval"]["start_seconds"]["d"] == "30000"
    assert event["needs_review"] is True
    assert result["long_run_analysis"]["max_buffer_frames"] <= TD.LONG_MAX_FRAMES
    assert event["metrics"]["window_clipped"] is False


@pytest.mark.parametrize("nonlinear,inverted", [(True, False), (False, True)])
def test_long_nonlinear_or_inverted_mixture_is_observed(nonlinear, inverted):
    result = TD.discover_events(samples(crossfade(1000, nonlinear=nonlinear, inverted=inverted)))
    assert len(long_events(result)) == 1


@pytest.mark.parametrize("kind", ["uniform_brightness", "textured_brightness", "contrast", "pan", "cut"])
def test_long_motion_and_illumination_controls_do_not_create_dissolves(kind):
    first, last = textures()
    if kind == "uniform_brightness":
        images = (np.full(first.shape, value) for value in np.linspace(30, 210, 1000))
    elif kind == "textured_brightness":
        images = (first + value for value in np.linspace(0, 30, 1000))
    elif kind == "contrast":
        images = (first * value for value in np.linspace(.65, 1.1, 1000))
    elif kind == "cut":
        images = itertools.chain([first] * 400, [last] * 400)
    else:
        y, x = np.indices(first.shape)
        images = (np.where(x < 8 + i / 12, 225, 25) + (y % 3) * 5 for i in range(500))
    result = TD.discover_events(samples(images))
    assert not long_events(result)
    assert not [e for e in result["events"] if e["kind"] == "dissolve"]


def test_range_ending_during_long_mixture_is_explicitly_clipped():
    images = itertools.islice(crossfade(1000), 850)
    event, = long_events(TD.discover_events(samples(images)))
    assert event["metrics"]["window_clipped"] is True
    assert event["interval"]["end_frame"] == "849"


@pytest.mark.parametrize("leading_hold", [0, 8])
def test_selected_start_with_short_plateau_has_no_unobserved_previous_frame(leading_hold):
    images = itertools.islice(crossfade(1000, holds=leading_hold), 850)
    event, = long_events(TD.discover_events(samples(images, offset=700)))
    assert event["metrics"]["window_clipped"] is True
    assert event["previous_frame"] is None
    assert int(event["interval"]["start_frame"]) >= 700
    assert event["interval"]["end_frame"] == "1549"


def test_normal_closed_long_run_uses_an_actually_observed_previous_frame():
    event, = long_events(TD.discover_events(samples(crossfade(1000), offset=700)))
    assert event["metrics"]["window_clipped"] is False
    assert int(event["previous_frame"]) == int(event["interval"]["start_frame"]) - 1
    assert int(event["previous_frame"]) >= 700


def test_admitted_frame_gap_ends_the_run_and_never_invents_missing_frames():
    frames = list(samples(crossfade(1000)))
    for frame in frames[600:]:
        frame["frame_index"] = str(int(frame["frame_index"]) + 50)
    result = TD.discover_events(frames)
    assert result["long_run_analysis"]["frame_gaps"] == 1
    for event in long_events(result):
        left, right = map(int, (event["interval"]["start_frame"], event["interval"]["end_frame"]))
        assert right < 600 or left >= 650
        assert event["metrics"]["window_clipped"] is True


def test_long_feature_buffer_budget_bounds_a_run_that_exceeds_the_limit():
    rng = np.random.default_rng(20261113)
    first = rng.integers(0, 256, size=(48, 64)).astype(float)
    last = np.flipud(first)
    images = ((1 - a) * first + a * last for a in np.linspace(0, 1, 2300))
    result = TD.discover_events(samples(images))
    assert result["long_run_analysis"]["max_buffer_frames"] <= TD.LONG_MAX_FRAMES
    assert result["long_run_analysis"]["budget_windows"] >= 1
    assert all(event["metrics"]["window_clipped"] for event in long_events(result))


def test_cancellation_inside_a_long_run_retains_control():
    observed = []
    def cancel():
        observed.append(1)
        return len(observed) >= 700
    with pytest.raises(WorkspaceError) as error:
        TD.discover_events(samples(crossfade(2000)), cancelled=cancel)
    assert error.value.code == "CANCELLED"
    assert len(observed) == 700


def test_scan_time_budget_is_checked_before_a_decoder_or_store_is_used(tmp_path, monkeypatch):
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(TD.time, "monotonic", lambda: next(clock))
    with pytest.raises(WorkspaceError) as error:
        TD.scan_transitions(None, "no-store-required", tmp_path / "timed-out", timeout=1)
    assert error.value.code == "FRAME_DECODE_TIMEOUT"


def test_long_candidate_configuration_exposes_its_measured_limits():
    config = TD.configuration()["long_dissolve"]
    assert config["max_frames"] == 2048
    assert config["max_static_frames"] == 16
    assert config["min_frames"] > TD.WINDOW_FRAMES
    assert config["image_precision"] == "rounded_uint8"
    assert config["max_split_attempts"] == 4
    assert config["endpoint_monotone_residual"] == .02
    assert config["filter_scope"] == "all_global_dissolves"


def test_nonlinear_gamma_change_keeps_its_competing_measurement_and_is_not_a_dissolve():
    y, x = np.indices((48, 64))
    original = ((x * 3 + y * 7) % 210 + 20).astype(float) / 255
    images = ((original ** gamma) * 255 for gamma in np.linspace(.5, 2.5, 1000))
    result = TD.discover_events(samples(images))
    assert not long_events(result)
    assert not [event for event in result["events"] if event["kind"] == "dissolve"]
    assert sum(result["long_run_analysis"]["short_dissolve_rejections"].values()) >= 1
    assert any(item["source_scope"] == "short_global_window"
               for item in result["long_run_analysis"]["rejected_fit_measurements"])
    rejection = next(item for item in result["long_run_analysis"]["rejected_fit_measurements"]
                     if item["reason"] == "monotone_luminance_mapping")
    assert rejection["endpoint_affine_residual"] > TD.LONG_ENDPOINT_AFFINE_RESIDUAL
    assert rejection["endpoint_monotone_residual"] < TD.LONG_ENDPOINT_MONOTONE_RESIDUAL
    assert len(result["long_run_analysis"]["rejected_fit_measurements"]) <= 16


@pytest.mark.parametrize("middle_hold", [8, 24])
def test_adjacent_long_mixtures_are_individually_fitted_across_a_real_plateau(middle_hold):
    first, middle = textures()
    last = np.flipud(first)
    first_mix = ((1 - a) * first + a * middle for a in np.linspace(0, 1, 700))
    second_mix = ((1 - a) * middle + a * last for a in np.linspace(0, 1, 700))
    images = itertools.chain([first] * 24, first_mix, [middle] * middle_hold, second_mix, [last] * 24)
    result = TD.discover_events(samples(images))
    events = long_events(result)
    assert len(events) == 2
    assert int(events[0]["interval"]["end_frame"]) < int(events[1]["interval"]["start_frame"])
    assert all(event["metrics"]["fit_interval"] == {
        "start_frame": event["interval"]["start_frame"], "end_frame": event["interval"]["end_frame"]}
               for event in events)
    if middle_hold == 8:
        assert result["long_run_analysis"]["split_accepted_pairs"] == 1
        assert 1 <= result["long_run_analysis"]["split_attempts"] <= 4
        assert result["long_run_analysis"]["rejected_runs"]
        assert all(event["metrics"]["measurement_method"] == "split_long_run_mixture" for event in events)
        assert all(event["metrics"]["split_platform"]["frames"] >= middle_hold for event in events)
    else:
        assert result["long_run_analysis"]["split_attempts"] == 0


def test_frozen_and_current_detector_configurations_remain_readable():
    current = TD.configuration()
    assert TD.validate_configuration(current) == current
    legacy = TD.configuration()
    legacy.pop("long_dissolve")
    assert TD.validate_configuration(legacy) == legacy
    checked = TD.validate_configuration(current)
    checked["long_dissolve"]["max_frames"] = 1
    assert TD.configuration()["long_dissolve"]["max_frames"] == 2048


@pytest.mark.parametrize("change", ["max_bytes", "long_dissolve", "unknown"])
def test_changed_or_unknown_detector_configuration_is_rejected(change):
    value = TD.configuration()
    if change == "long_dissolve":
        value[change]["max_frames"] += 1
    else:
        value[change] = 1
    with pytest.raises(WorkspaceError) as error:
        TD.validate_configuration(value)
    assert error.value.code == "SCHEMA_INVALID"
