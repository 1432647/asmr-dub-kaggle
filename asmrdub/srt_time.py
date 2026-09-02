"""SRT / dataframe time-string conversion.

VideoLingo mixes three time formats:
  - SRT proper:            "00:01:02,500"  (comma)
  - tts_tasks.xlsx:        "00:01:02.500"  (dot)
  - internal float seconds

These helpers are the single place that knows about all three.
"""

from __future__ import annotations

import re

_TIME_RE = re.compile(r"^(\d+):([0-5]?\d):([0-5]?\d)(?:[.,](\d{1,6}))?$")


def parse_time(value: str | float | int) -> float:
    """Parse "H:MM:SS,mmm" / "H:MM:SS.mmm" / "H:MM:SS" into seconds.

    Floats and ints pass through, so callers can be sloppy about whether a
    dataframe column has already been converted.
    """
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"unparseable time: {value!r}")
    hours, minutes, seconds, frac = match.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    if frac:
        total += int(frac) / (10 ** len(frac))
    return total


def format_time(seconds: float, sep: str = ",") -> str:
    """Format seconds as "HH:MM:SS<sep>mmm", clamping negatives to zero.

    Milliseconds are truncated rather than rounded so a formatted timestamp
    never lands past the sample it describes.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(seconds * 1000 + 1e-6)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    return "%02d:%02d:%02d%s%03d" % (
        total_s // 3600,
        (total_s % 3600) // 60,
        total_s % 60,
        sep,
        ms,
    )


def format_srt(seconds: float) -> str:
    """Format for .srt files (comma decimal separator)."""
    return format_time(seconds, sep=",")


def format_df(seconds: float) -> str:
    """Format for tts_tasks.xlsx columns (dot decimal separator)."""
    return format_time(seconds, sep=".")


def build_srt(entries: list[tuple[float, float, str]]) -> str:
    """Render (start, end, text) triples as SRT text.

    Empty-text entries are dropped; zero-or-negative-length entries are
    stretched to 1ms so players do not skip them.
    """
    blocks = []
    index = 0
    for start, end, text in entries:
        text = (text or "").strip()
        if not text:
            continue
        if end <= start:
            end = start + 0.001
        index += 1
        blocks.append(
            "%d\n%s --> %s\n%s\n" % (index, format_srt(start), format_srt(end), text)
        )
    return "\n".join(blocks)
