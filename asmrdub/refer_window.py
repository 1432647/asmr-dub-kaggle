"""Reference-clip window expansion for voice cloning.

Every dubbed line is cloned from the *original* line's audio, so the speaker
follows the source automatically -- no diarization needed. The catch is short
lines: IndexTTS needs a few seconds of reference to lock onto a timbre, and a
0.8s "...ん?" produces a wandering voice.

`expand_window` grows a line's window by absorbing neighbours, but only across
short silences. A long gap usually means the scene or the speaker changed, and
absorbing across it would blend two voices into one reference -- worse than a
short clip.

Pure functions over plain tuples: no audio, no torch, fully unit-testable.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


class Span(NamedTuple):
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def expand_window(
    spans: Sequence[tuple[float, float]],
    index: int,
    min_sec: float = 3.0,
    max_sec: float = 10.0,
    join_gap: float = 1.2,
    total_duration: float | None = None,
) -> Span:
    """Grow ``spans[index]`` until it is at least ``min_sec`` long.

    Args:
        spans: line windows as (start, end) seconds, ascending by start.
        index: which line to build a reference clip for.
        min_sec: stop growing once the window reaches this length.
        max_sec: never exceed this length.
        join_gap: only absorb a neighbour whose silence gap is <= this.
        total_duration: hard ceiling for the returned end, if known.

    Returns:
        The expanded window. Never shorter than the seed span; may still be
        shorter than ``min_sec`` when no neighbour is close enough.

    Growth alternates sides, always taking the side with the smaller gap, so
    the reference stays centred on the seed line rather than drifting.

    ``max_sec`` always wins over ``min_sec``: a config with min > max is
    contradictory but its intent is unambiguous, and the hard ceiling is the
    one the model actually cares about.
    """
    if not spans:
        raise ValueError("spans is empty")
    if not 0 <= index < len(spans):
        raise IndexError(f"index {index} out of range for {len(spans)} spans")
    if min_sec < 0 or max_sec <= 0:
        raise ValueError("min_sec must be >= 0 and max_sec > 0")
    min_sec = min(float(min_sec), float(max_sec))

    ordered = [Span(float(s), float(e)) for s, e in spans]
    seed = ordered[index]
    start, end = seed.start, seed.end
    lo = hi = index

    while (end - start) < min_sec:
        prev_gap = start - ordered[lo - 1].end if lo > 0 else None
        next_gap = ordered[hi + 1].start - end if hi + 1 < len(ordered) else None

        can_prev = prev_gap is not None and prev_gap <= join_gap
        can_next = next_gap is not None and next_gap <= join_gap
        if not can_prev and not can_next:
            break

        # Prefer the tighter gap: more likely the same speaker continuing.
        take_prev = can_prev and (not can_next or prev_gap <= next_gap)
        candidate_start, candidate_end = start, end
        if take_prev:
            candidate_start = ordered[lo - 1].start
        else:
            candidate_end = ordered[hi + 1].end

        if (candidate_end - candidate_start) > max_sec:
            # Absorbing the whole neighbour overshoots; take only the slice we
            # need from its far edge and stop.
            need = max_sec - (end - start)
            if need > 0:
                if take_prev:
                    start = max(candidate_start, start - need)
                else:
                    end = min(candidate_end, end + need)
            break

        start, end = candidate_start, candidate_end
        if take_prev:
            lo -= 1
        else:
            hi += 1

    if start < 0:
        start = 0.0
    if total_duration is not None and end > total_duration:
        end = float(total_duration)
        if end < start:
            end = start
    return Span(start, end)


def pick_fallback_index(
    spans: Sequence[tuple[float, float]], min_sec: float = 3.0
) -> int | None:
    """Index of the longest span, used when a line has no usable reference.

    Returns None when even the longest span is under ``min_sec`` -- the caller
    should then just use whatever it has rather than substituting a clip from
    a different moment.
    """
    if not spans:
        return None
    best = max(range(len(spans)), key=lambda i: spans[i][1] - spans[i][0])
    if (spans[best][1] - spans[best][0]) < min_sec:
        return None
    return best
