import math

import pytest

from asmrdub.chunking import (
    Chunk,
    fade_envelope,
    fade_gain,
    plan_chunks,
    verify_coverage,
)

SR = 44100


def test_short_audio_is_one_chunk_without_fades():
    chunks = plan_chunks(1000, SR, chunk_sec=10.0)
    assert chunks == [Chunk(0, 0, 1000, 0, 0)]


def test_empty_audio_plans_nothing():
    assert plan_chunks(0, SR) == []


def test_long_audio_tiles_completely():
    total = SR * 60 * 15  # 15 minutes
    chunks = plan_chunks(total, SR, chunk_sec=10.0, overlap=0.1)
    assert len(chunks) > 1
    assert verify_coverage(chunks, total)
    assert chunks[0].start == 0
    assert chunks[-1].end == total


def test_edges_keep_unit_gain():
    chunks = plan_chunks(SR * 100, SR, chunk_sec=10.0, overlap=0.1)
    assert chunks[0].fade_in == 0
    assert chunks[-1].fade_out == 0
    for chunk in chunks[1:-1]:
        assert chunk.fade_in > 0
        assert chunk.fade_out > 0


def test_overlap_equals_one_fade():
    """Chunk i's fade-out region is exactly chunk i+1's fade-in region."""
    chunks = plan_chunks(SR * 100, SR, chunk_sec=10.0, overlap=0.1)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.end - current.start == current.fade_in
        assert previous.fade_out == current.fade_in


def test_fades_never_overlap_each_other():
    chunks = plan_chunks(SR * 100, SR, chunk_sec=10.0, overlap=0.1)
    for chunk in chunks:
        assert chunk.fade_in + chunk.fade_out <= chunk.length
        fade_envelope(chunk)  # would raise otherwise


def test_seam_gains_sum_to_exactly_one():
    chunks = plan_chunks(SR * 100, SR, chunk_sec=10.0, overlap=0.1)
    for previous, current in zip(chunks, chunks[1:]):
        fade = current.fade_in
        for offset in range(fade):
            position = current.start + offset
            left = fade_gain(previous, position - previous.start)
            right = fade_gain(current, offset)
            assert left + right == pytest.approx(1.0, abs=1e-12)


def test_faded_sum_reconstructs_constant_signal():
    """A DC signal must come back as DC after fade-and-sum."""
    total = SR * 45
    chunks = plan_chunks(total, SR, chunk_sec=10.0, overlap=0.1)
    canvas = [0.0] * total
    for chunk in chunks:
        for offset, gain in enumerate(fade_envelope(chunk)):
            canvas[chunk.start + offset] += gain
    for value in canvas:
        assert math.isclose(value, 1.0, abs_tol=1e-9)


def test_no_tiny_trailing_chunk():
    """A sliver of a final chunk wastes a pass and puts artefacts in-band."""
    total = SR * 41  # 4 hops of 10s plus a 1s tail
    chunks = plan_chunks(total, SR, chunk_sec=10.0, overlap=0.1)
    assert verify_coverage(chunks, total)
    assert chunks[-1].length >= SR * 5


def test_zero_overlap_still_tiles():
    total = SR * 35
    chunks = plan_chunks(total, SR, chunk_sec=10.0, overlap=0.0)
    assert verify_coverage(chunks, total)
    assert all(c.fade_in == 0 and c.fade_out == 0 for c in chunks)


def test_zero_overlap_reconstructs_exactly():
    total = SR * 35
    chunks = plan_chunks(total, SR, chunk_sec=10.0, overlap=0.0)
    canvas = [0.0] * total
    for chunk in chunks:
        for offset, gain in enumerate(fade_envelope(chunk)):
            canvas[chunk.start + offset] += gain
    assert all(v == 1.0 for v in canvas)


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        plan_chunks(1000, 0)
    with pytest.raises(ValueError):
        plan_chunks(1000, SR, overlap=0.5)


def test_fade_gain_rejects_out_of_range_offset():
    chunk = Chunk(0, 0, 10, 0, 0)
    with pytest.raises(IndexError):
        fade_gain(chunk, 10)


def test_fade_envelope_rejects_overlapping_fades():
    with pytest.raises(ValueError, match="exceed length"):
        fade_envelope(Chunk(1, 0, 10, 6, 6))


def test_verify_coverage_detects_gap():
    holed = [Chunk(0, 0, 100, 0, 0), Chunk(1, 150, 300, 0, 0)]
    assert not verify_coverage(holed, 300)
