"""Overlap-add chunk planning for long-audio source separation.

HDemucs on a 15-minute 44.1kHz stereo file is ~40M samples per channel and
will not fit in a T4. The standard fix (torchaudio's own tutorial) is to run
fixed-length chunks with a small overlap and cross-fade the seams, which
hides the model's edge artefacts.

The invariant that matters: the fade windows must sum to exactly 1 across
every seam, or the crossfade region comes out louder (or quieter) than the
rest of the track. That is easy to get subtly wrong, so the envelope lives
here next to the plan and both are unit-tested against a DC signal -- no GPU
and no audio needed.

Layout: chunk *i* starts at ``i * hop`` and is ``hop + fade`` samples long, so
consecutive chunks overlap by exactly ``fade`` -- the same length as the
fade-out of the earlier chunk and the fade-in of the later one.
"""

from __future__ import annotations

from typing import NamedTuple


class Chunk(NamedTuple):
    index: int
    start: int          # inclusive, in samples, into the source
    end: int            # exclusive
    fade_in: int        # samples of fade-in applied to this chunk
    fade_out: int       # samples of fade-out applied to this chunk

    @property
    def length(self) -> int:
        return self.end - self.start


def plan_chunks(
    total_samples: int,
    sample_rate: int,
    chunk_sec: float = 10.0,
    overlap: float = 0.1,
) -> list[Chunk]:
    """Split ``total_samples`` into overlapping chunks with fade lengths.

    Args:
        total_samples: length of the source in samples.
        sample_rate: samples per second.
        chunk_sec: hop length in seconds; each chunk is this plus the overlap.
        overlap: overlap as a fraction of ``chunk_sec``, in [0, 0.5).

    Returns:
        Chunks in order. Applying ``fade_envelope`` to each and summing them at
        their ``start`` offsets reconstructs a signal of exactly
        ``total_samples`` with unit gain everywhere.

    The first chunk has no fade-in and the last no fade-out, so the very
    beginning and end of the track keep unit gain. A short final chunk is
    absorbed into its predecessor rather than emitted on its own -- running the
    model on a 0.2s tail wastes a forward pass and its edge artefacts would
    land inside the audible region.
    """
    if total_samples <= 0:
        return []
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 <= overlap < 0.5:
        raise ValueError("overlap must be in [0, 0.5)")

    hop = max(1, int(round(chunk_sec * sample_rate)))
    fade = int(round(hop * overlap))
    if fade * 2 >= hop:
        fade = max(0, (hop - 1) // 2)

    if total_samples <= hop + fade:
        return [Chunk(0, 0, total_samples, 0, 0)]

    chunks: list[Chunk] = []
    start = 0
    index = 0
    min_tail = hop // 2
    while True:
        end = start + hop + fade
        if total_samples - end < min_tail:
            end = total_samples
        end = min(end, total_samples)
        is_last = end >= total_samples
        chunks.append(
            Chunk(
                index=index,
                start=start,
                end=end,
                fade_in=fade if index > 0 else 0,
                fade_out=0 if is_last else fade,
            )
        )
        if is_last:
            break
        start += hop
        index += 1
    return chunks


def fade_gain(chunk: Chunk, offset: int) -> float:
    """Gain to apply to ``chunk`` at local sample ``offset``.

    Fade-in rises 0 -> (fade-1)/fade and the matching fade-out falls
    1 -> 1/fade, so the two sum to exactly 1 at every position in the seam.
    """
    length = chunk.length
    if not 0 <= offset < length:
        raise IndexError(f"offset {offset} out of range for length {length}")
    gain = 1.0
    if chunk.fade_in and offset < chunk.fade_in:
        gain *= offset / chunk.fade_in
    if chunk.fade_out:
        tail_start = length - chunk.fade_out
        if offset >= tail_start:
            gain *= 1.0 - (offset - tail_start) / chunk.fade_out
    return gain


def fade_envelope(chunk: Chunk) -> list[float]:
    """The full per-sample gain envelope for ``chunk``.

    Raises when the fades would overlap each other, which would make the
    envelope multiplicative in the middle and break the sum-to-one property.
    """
    if chunk.fade_in + chunk.fade_out > chunk.length:
        raise ValueError(
            f"chunk {chunk.index}: fades ({chunk.fade_in}+{chunk.fade_out}) "
            f"exceed length {chunk.length}"
        )
    return [fade_gain(chunk, i) for i in range(chunk.length)]


def verify_coverage(chunks: list[Chunk], total_samples: int) -> bool:
    """True when the chunks tile [0, total_samples) with no hole.

    A gap would silently drop audio, so callers assert on this before
    spending GPU time.
    """
    if not chunks:
        return total_samples == 0
    if chunks[0].start != 0:
        return False
    if chunks[-1].end != total_samples:
        return False
    reach = chunks[0].end
    for chunk in chunks[1:]:
        if chunk.start > reach:
            return False
        reach = max(reach, chunk.end)
    return reach == total_samples
