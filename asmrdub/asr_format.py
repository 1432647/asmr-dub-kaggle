"""Convert faster-whisper output into the shape VideoLingo expects.

VideoLingo's ``core/asr_backend/audio_preprocess.process_transcription`` wants

    {"segments": [{"start", "end", "text",
                   "words": [{"word", "start", "end"}, ...]}, ...]}

which is whisperX's shape. faster-whisper yields dataclass-ish objects with
``.words`` whose entries also use ``.word``. This module does the translation
and, importantly, repairs missing/degenerate word timings -- downstream code
indexes ``word['end']`` unconditionally and a None there crashes the run
20 minutes in.
"""

from __future__ import annotations

from typing import Any, Iterable


def _attr(obj: Any, name: str, default=None):
    """Read ``name`` from an object or a dict, whichever we were handed."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def words_from_segment(segment: Any) -> list[dict]:
    """Extract word dicts, synthesising one from the segment when absent.

    faster-whisper omits ``words`` when word_timestamps=False, and can emit an
    empty list for a segment of pure non-speech. Either way we still need one
    entry so sentence-level alignment has something to anchor to.
    """
    raw_words = _attr(segment, "words", None) or []
    seg_start = _attr(segment, "start", 0.0)
    seg_end = _attr(segment, "end", seg_start)
    words: list[dict] = []
    for raw in raw_words:
        text = (_attr(raw, "word", "") or "")
        if not text.strip():
            continue
        start = _attr(raw, "start", None)
        end = _attr(raw, "end", None)
        words.append(
            {
                "word": text,
                "start": float(start) if start is not None else None,
                "end": float(end) if end is not None else None,
            }
        )
    if not words:
        text = (_attr(segment, "text", "") or "").strip()
        if not text:
            return []
        return [
            {"word": text, "start": float(seg_start), "end": float(seg_end)}
        ]
    return words


def repair_word_times(words: list[dict], seg_start: float, seg_end: float) -> list[dict]:
    """Fill in None timings and enforce a non-decreasing timeline.

    A word with no timing inherits the previous word's end (zero-length), and
    any end that precedes its own start is clamped forward. VideoLingo builds
    subtitle spans by indexing the first and last word of a sentence, so a
    single inverted pair produces a negative-duration subtitle.
    """
    cursor = float(seg_start)
    for word in words:
        start = word.get("start")
        end = word.get("end")
        if start is None:
            start = cursor
        if end is None:
            end = start
        start = max(float(start), cursor)
        end = max(float(end), start)
        word["start"] = start
        word["end"] = end
        cursor = end
    if words:
        words[-1]["end"] = max(words[-1]["end"], min(float(seg_end), words[-1]["end"]))
    return words


def to_videolingo(
    segments: Iterable[Any],
    language: str,
    time_offset: float = 0.0,
) -> dict:
    """Build a VideoLingo-shaped result dict.

    Args:
        segments: faster-whisper segments (objects or dicts).
        language: detected language code, e.g. "ja".
        time_offset: added to every timestamp; used when the source audio was
            split into pieces and each piece transcribed independently.

    Empty-text segments are dropped: they carry no words for alignment and
    would produce empty subtitle rows.
    """
    out_segments = []
    for segment in segments:
        text = (_attr(segment, "text", "") or "").strip()
        seg_start = float(_attr(segment, "start", 0.0) or 0.0)
        seg_end = float(_attr(segment, "end", seg_start) or seg_start)
        words = words_from_segment(segment)
        if not text and not words:
            continue
        words = repair_word_times(words, seg_start, seg_end)
        if time_offset:
            seg_start += time_offset
            seg_end += time_offset
            for word in words:
                word["start"] += time_offset
                word["end"] += time_offset
        out_segments.append(
            {
                "start": seg_start,
                "end": seg_end,
                "text": text or "".join(w["word"] for w in words),
                "words": words,
            }
        )
    return {"segments": out_segments, "language": language}


def merge_results(results: Iterable[dict]) -> dict:
    """Concatenate several to_videolingo() dicts in chronological order."""
    merged: list[dict] = []
    language = None
    for result in results:
        if language is None:
            language = result.get("language")
        merged.extend(result.get("segments", []))
    merged.sort(key=lambda s: s["start"])
    return {"segments": merged, "language": language or "ja"}
