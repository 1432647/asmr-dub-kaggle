"""Stereo mixdown maths for ASMR dubbing.

ASMR is usually binaural: the performer moves between the ears, and that
movement *is* the content. A mono dub dropped in the centre destroys it. So
for every line we measure where the original voice sat in the stereo field
and place the Chinese dub at the same spot.

Pure numeric helpers only -- no numpy, no torch, no file I/O -- so the panning
law and the gain staging are unit-testable.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence


class PanGains(NamedTuple):
    left: float
    right: float


def pan_from_energy(left_energy: float, right_energy: float) -> float:
    """Estimate a pan position in [-1, 1] from per-channel energy.

    -1 is hard left, 0 centre, +1 hard right. Energies are sums of squared
    samples (or mean squares -- only their ratio matters).

    This is the exact inverse of ``equal_power_gains``: measuring a voice's
    position and then re-placing a mono dub there reproduces the original
    channel balance. Getting that round-trip right is the entire point of
    measuring, so the two functions must share one pan law -- a
    normalised-amplitude-difference estimate looks reasonable in isolation but
    lands a hard-left voice around -0.5.
    """
    left_energy = max(0.0, float(left_energy))
    right_energy = max(0.0, float(right_energy))
    if left_energy + right_energy <= 0:
        return 0.0
    angle = math.atan2(math.sqrt(right_energy), math.sqrt(left_energy))  # 0 .. pi/2
    return max(-1.0, min(1.0, angle * (4.0 / math.pi) - 1.0))


def equal_power_gains(pan: float) -> PanGains:
    """Constant-power pan law: gains for a mono source at ``pan``.

    left^2 + right^2 == 1 for every pan, so perceived loudness stays constant
    as the voice moves. At pan=0 both gains are 1/sqrt(2) ~ 0.7071.
    """
    pan = max(-1.0, min(1.0, float(pan)))
    angle = (pan + 1.0) * (math.pi / 4.0)  # 0 -> pi/2
    return PanGains(left=math.cos(angle), right=math.sin(angle))


def limit_pan(pan: float, max_spread: float = 0.9) -> float:
    """Scale a pan estimate toward centre.

    Energy-based estimates on a de-mixed vocal track are noisy; hard-panning
    on a bad estimate is far more audible than under-panning, so we keep a
    little headroom on both sides.
    """
    max_spread = max(0.0, min(1.0, float(max_spread)))
    return max(-1.0, min(1.0, float(pan))) * max_spread


class Placement(NamedTuple):
    number: int
    line_index: int
    start_sample: int
    pan: float
    gain_left: float
    gain_right: float


def plan_placements(
    segments: Sequence[dict],
    sample_rate: int,
    max_spread: float = 0.9,
) -> list[Placement]:
    """Turn per-line timing + pan into sample offsets and channel gains.

    Each segment dict needs: ``number``, ``line_index``, ``start`` (seconds)
    and ``pan`` in [-1, 1]. A missing or None pan means centre.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    placements = []
    for seg in segments:
        pan = limit_pan(seg.get("pan") or 0.0, max_spread)
        gains = equal_power_gains(pan)
        start = max(0.0, float(seg["start"]))
        placements.append(
            Placement(
                number=int(seg["number"]),
                line_index=int(seg.get("line_index", 0)),
                start_sample=int(round(start * sample_rate)),
                pan=pan,
                gain_left=gains.left,
                gain_right=gains.right,
            )
        )
    placements.sort(key=lambda p: (p.start_sample, p.number, p.line_index))
    return placements


def canvas_length(
    placements: Sequence[Placement],
    segment_lengths: dict[tuple[int, int], int],
    background_samples: int,
    tail_samples: int = 0,
) -> int:
    """Length of the output canvas in samples.

    Must cover the background bed and every placed dub -- a dub that starts
    near the end can run past the original's duration once it has been
    time-stretched.
    """
    longest = background_samples
    for placement in placements:
        length = segment_lengths.get((placement.number, placement.line_index), 0)
        longest = max(longest, placement.start_sample + length)
    return max(0, longest + max(0, tail_samples))


def peak_normalize_gain(peak: float, ceiling: float = 0.99) -> float:
    """Gain that brings ``peak`` down to ``ceiling``; 1.0 when already under.

    Deliberately a flat scale, not a compressor: ASMR dynamics are part of the
    experience and compressing them would flatten exactly what listeners want.
    """
    peak = abs(float(peak))
    if peak <= ceiling or peak == 0.0:
        return 1.0
    return ceiling / peak
