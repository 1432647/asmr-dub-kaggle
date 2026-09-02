import math

import pytest

from asmrdub.mixdown import (
    canvas_length,
    equal_power_gains,
    limit_pan,
    pan_from_energy,
    peak_normalize_gain,
    plan_placements,
)


def test_balanced_energy_is_centre():
    assert pan_from_energy(1.0, 1.0) == pytest.approx(0.0)


def test_right_only_is_hard_right():
    assert pan_from_energy(0.0, 1.0) == pytest.approx(1.0)


def test_left_only_is_hard_left():
    assert pan_from_energy(1.0, 0.0) == pytest.approx(-1.0)


def test_silence_is_centre_not_nan():
    assert pan_from_energy(0.0, 0.0) == 0.0


def test_estimate_is_the_inverse_of_the_pan_law():
    """Measure-then-place must be a round trip, or the dub moves ears."""
    for pan in (-1.0, -0.8, -0.35, 0.0, 0.35, 0.8, 1.0):
        gains = equal_power_gains(pan)
        recovered = pan_from_energy(gains.left**2, gains.right**2)
        assert recovered == pytest.approx(pan, abs=1e-9)


def test_pan_estimate_is_monotonic():
    values = [pan_from_energy(1.0, r) for r in (0.25, 0.5, 1.0, 2.0, 4.0)]
    assert values == sorted(values)


def test_equal_power_law_preserves_power():
    for pan in (-1.0, -0.5, 0.0, 0.25, 1.0):
        gains = equal_power_gains(pan)
        assert gains.left**2 + gains.right**2 == pytest.approx(1.0)


def test_centre_gain_is_root_half():
    gains = equal_power_gains(0.0)
    assert gains.left == pytest.approx(1 / math.sqrt(2))
    assert gains.right == pytest.approx(1 / math.sqrt(2))


def test_hard_pans_are_one_and_zero():
    left = equal_power_gains(-1.0)
    assert left.left == pytest.approx(1.0)
    assert left.right == pytest.approx(0.0, abs=1e-12)
    right = equal_power_gains(1.0)
    assert right.right == pytest.approx(1.0)


def test_out_of_range_pan_is_clamped():
    assert equal_power_gains(5.0) == equal_power_gains(1.0)
    assert equal_power_gains(-5.0) == equal_power_gains(-1.0)


def test_limit_pan_keeps_headroom():
    assert limit_pan(1.0, 0.9) == pytest.approx(0.9)
    assert limit_pan(-1.0, 0.9) == pytest.approx(-0.9)
    assert limit_pan(0.0, 0.9) == 0.0


def test_placements_are_sorted_and_converted():
    segments = [
        {"number": 2, "line_index": 0, "start": 1.0, "pan": 0.5},
        {"number": 1, "line_index": 0, "start": 0.0, "pan": None},
    ]
    placements = plan_placements(segments, 44100)
    assert [p.number for p in placements] == [1, 2]
    assert placements[0].start_sample == 0
    assert placements[1].start_sample == 44100
    assert placements[0].pan == 0.0


def test_negative_start_clamped_to_zero():
    placements = plan_placements(
        [{"number": 1, "line_index": 0, "start": -3.0, "pan": 0.0}], 44100
    )
    assert placements[0].start_sample == 0


def test_canvas_covers_dub_running_past_background():
    placements = plan_placements(
        [{"number": 1, "line_index": 0, "start": 10.0, "pan": 0.0}], 100
    )
    lengths = {(1, 0): 500}
    assert canvas_length(placements, lengths, background_samples=1200) == 1500


def test_canvas_covers_background_when_longer():
    placements = plan_placements(
        [{"number": 1, "line_index": 0, "start": 0.0, "pan": 0.0}], 100
    )
    assert canvas_length(placements, {(1, 0): 10}, background_samples=9999) == 9999


def test_canvas_adds_tail():
    assert canvas_length([], {}, background_samples=100, tail_samples=50) == 150


def test_peak_normalize_only_when_clipping():
    assert peak_normalize_gain(0.5) == 1.0
    assert peak_normalize_gain(0.0) == 1.0
    assert peak_normalize_gain(2.0, ceiling=1.0) == pytest.approx(0.5)
    assert peak_normalize_gain(-2.0, ceiling=1.0) == pytest.approx(0.5)


def test_rejects_bad_sample_rate():
    with pytest.raises(ValueError):
        plan_placements([], 0)
