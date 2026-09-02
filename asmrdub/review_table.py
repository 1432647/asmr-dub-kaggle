"""Review-table round-trip between tts_tasks.xlsx and the Streamlit editor.

Japanese ASMR is the worst case for ASR: whispering, breath, trailing
particles. Rather than pretend otherwise, the UI shows the recognised
Japanese next to the Chinese translation and lets the user fix both before
any TTS runs.

The functions here convert between the on-disk task rows and the flat table
the editor shows, and validate edits. They operate on lists of plain dicts so
pandas is not needed for the tests.
"""

from __future__ import annotations

from typing import Any, Iterable

from asmrdub.srt_time import format_df, parse_time

EDITOR_COLUMNS = ("number", "start", "end", "duration", "origin", "text")


def to_editor_rows(rows: Iterable[dict]) -> list[dict]:
    """Project task rows onto the editable columns.

    ``duration`` is recomputed from start/end rather than trusted: upstream
    writes it once and later stages mutate end_time.
    """
    out = []
    for row in rows:
        start = parse_time(row["start_time"])
        end = parse_time(row["end_time"])
        out.append(
            {
                "number": int(row["number"]),
                "start": format_df(start),
                "end": format_df(end),
                "duration": round(end - start, 3),
                "origin": _as_text(row.get("origin")),
                "text": _as_text(row.get("text")),
            }
        )
    return out


def apply_editor_rows(rows: Iterable[dict], edited: Iterable[dict]) -> list[dict]:
    """Merge edited text back into the task rows, keyed by ``number``.

    Only ``origin`` and ``text`` are writable. Timings stay authoritative on
    disk, because they came from forced alignment and the chunking pass has
    already reasoned about the gaps between them.
    """
    by_number = {int(item["number"]): item for item in edited}
    merged = []
    for row in rows:
        number = int(row["number"])
        update = by_number.get(number)
        new_row = dict(row)
        if update is not None:
            if "text" in update:
                new_row["text"] = _as_text(update["text"])
            if "origin" in update:
                new_row["origin"] = _as_text(update["origin"])
        merged.append(new_row)
    return merged


def validate_editor_rows(edited: Iterable[dict]) -> list[str]:
    """Return human-readable problems; empty list means the edit is usable.

    Blank translations are an error rather than a silent skip: upstream would
    substitute 100ms of silence and the line would just vanish from the dub.
    """
    problems = []
    seen: set[int] = set()
    for position, item in enumerate(edited, start=1):
        try:
            number = int(item["number"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"第 {position} 行缺少有效的 number")
            continue
        if number in seen:
            problems.append(f"number={number} 重复")
        seen.add(number)
        text = _as_text(item.get("text"))
        if not text:
            problems.append(f"number={number} 的中文译文为空")
    if not seen:
        problems.append("复核表为空")
    return problems


def _as_text(value: Any) -> str:
    """Coerce a cell to stripped text, mapping pandas NaN to empty.

    NaN != NaN is the cheapest float-nan test that needs no numpy import.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()
