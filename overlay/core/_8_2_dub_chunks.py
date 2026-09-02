"""Replacement for VideoLingo's core/_8_2_dub_chunks.py.

Upstream keeps its own timing analysis (which we want) and then does something
we cannot keep: it re-matches every `tts_tasks.text` against the lines of
`output/trans.srt` by exact string concatenation, raising
`ValueError("Matching failed")` on any mismatch.

That exists only to split one dub row into several display-subtitle lines. It is
useless for dubbing -- and it makes manual review impossible, because the moment
a human fixes a translation the match breaks and the run dies.

So the timing analysis is imported unchanged from upstream and only the matching
step is replaced with a 1:1 mapping. One row, one line, no fragile string
comparison.
"""

from __future__ import annotations

import pandas as pd

from core._8_2_dub_chunks_upstream import (
    analyze_subtitle_timing_and_speed,
    process_cutoffs,
)
from core.utils import rprint
from core.utils.models import _8_1_AUDIO_TASK


def gen_dub_chunks():
    rprint("[🎬 Starting] Generating dubbing chunks ...")
    df = pd.read_excel(_8_1_AUDIO_TASK)

    # Upstream logic, untouched: gap/tolerance/est_dur/if_too_fast, then the
    # cut-off points that decide which rows share one speed factor.
    df = analyze_subtitle_timing_and_speed(df)
    df = process_cutoffs(df)

    # One task row == one synthesised clip.
    df["lines"] = [[_text(value)] for value in df["text"]]
    df["src_lines"] = [[_text(value)] for value in df.get("origin", df["text"])]

    df.to_excel(_8_1_AUDIO_TASK, index=False)
    fast = int((df["if_too_fast"] == 2).sum())
    rprint(
        "[✅ Complete] %d 段配音任务已生成%s"
        % (len(df), f"，其中 {fast} 段即使加速也塞不进原时间轴" if fast else "")
    )


def _text(value) -> str:
    """Stringify a cell, mapping pandas NaN to empty.

    NaN != NaN is the cheapest nan test that avoids importing numpy here.
    """
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


if __name__ == "__main__":
    gen_dub_chunks()
