import pytest

from asmrdub.refer_window import expand_window, pick_fallback_index


def test_long_enough_line_is_untouched():
    spans = [(0.0, 5.0), (6.0, 9.0)]
    assert expand_window(spans, 0, min_sec=3.0) == (0.0, 5.0)


def test_short_line_absorbs_close_neighbour():
    spans = [(0.0, 1.0), (1.3, 2.0), (10.0, 12.0)]
    window = expand_window(spans, 0, min_sec=3.0, join_gap=1.2)
    # grew right into span 1 (gap 0.3) but stopped at the 8s silence
    assert window.start == 0.0
    assert window.end == 2.0


def test_long_gap_blocks_expansion():
    """A big silence usually means a new speaker; blending would be worse."""
    spans = [(0.0, 0.8), (5.0, 9.0)]
    window = expand_window(spans, 0, min_sec=3.0, join_gap=1.2)
    assert window == (0.0, 0.8)


def test_prefers_the_tighter_gap():
    spans = [(0.0, 1.0), (1.1, 2.0), (2.9, 3.5)]
    # seed is index 1; left gap 0.1 < right gap 0.9 -> take left first
    window = expand_window(spans, 1, min_sec=1.5, join_gap=1.2)
    assert window.start == 0.0
    assert window.end == 2.0


def test_expands_both_directions_when_needed():
    spans = [(0.0, 0.6), (1.0, 1.6), (2.0, 2.6)]
    window = expand_window(spans, 1, min_sec=2.5, join_gap=1.2)
    assert window.start == 0.0
    assert window.end == 2.6


def test_never_exceeds_max_sec():
    """A contradictory min>max config must clamp, not raise mid-run."""
    spans = [(0.0, 6.0), (6.2, 7.0), (7.2, 13.0)]
    window = expand_window(spans, 1, min_sec=10.0, max_sec=8.0, join_gap=1.2)
    assert window.duration <= 8.0 + 1e-9
    assert window.duration >= 0.8


def test_partial_slice_when_neighbour_overshoots():
    spans = [(0.0, 1.0), (1.1, 30.0)]
    window = expand_window(spans, 0, min_sec=3.0, max_sec=5.0, join_gap=1.2)
    assert window.start == 0.0
    assert window.duration == pytest.approx(5.0)


def test_clamped_to_total_duration():
    spans = [(0.0, 1.0), (1.2, 4.0)]
    window = expand_window(spans, 0, min_sec=3.0, join_gap=1.2, total_duration=3.5)
    assert window.end == 3.5


def test_never_negative_start():
    window = expand_window([(0.0, 0.5)], 0, min_sec=3.0)
    assert window.start == 0.0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        expand_window([], 0)
    with pytest.raises(IndexError):
        expand_window([(0.0, 1.0)], 5)
    with pytest.raises(ValueError):
        expand_window([(0.0, 1.0)], 0, max_sec=0.0)


def test_fallback_picks_longest():
    assert pick_fallback_index([(0.0, 1.0), (2.0, 8.0), (9.0, 10.0)]) == 1


def test_fallback_none_when_all_short():
    assert pick_fallback_index([(0.0, 1.0), (2.0, 2.5)], min_sec=3.0) is None
    assert pick_fallback_index([]) is None
