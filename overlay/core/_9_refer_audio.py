"""Replacement for VideoLingo's core/_9_refer_audio.py.

Two changes, both about clone quality:

1. Reference clips are cut from the 44.1kHz stereo vocal stem, not the 16kHz
   mono mp3. IndexTTS resamples internally, but it cannot invent the high
   frequencies that carry breath and sibilance -- exactly the cues that make an
   ASMR voice recognisable.
2. Short lines get an expanded window. A 0.8s "...ん?" is not enough reference
   for the model to lock onto a timbre, so the window absorbs neighbouring
   lines across short silences only (see asmrdub.refer_window for why long gaps
   are refused).

Also writes `refers/index.json` recording, per line, the window actually used
and its pan position -- the mixdown stage reads the pan from there instead of
re-measuring, so the dub lands exactly where the original voice was.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from rich.panel import Panel

from core.asr_backend.demucs_vl import VOCAL_HIFI_FILE, demucs_audio
from core.utils import rprint
from core.utils.models import _8_1_AUDIO_TASK, _AUDIO_REFERS_DIR

sys.path.insert(0, os.environ.get("ASMRDUB_PKG_PATH", ""))
from asmrdub.refer_window import expand_window, pick_fallback_index  # noqa: E402
from asmrdub.mixdown import pan_from_energy  # noqa: E402
from asmrdub.srt_time import parse_time  # noqa: E402

REFER_INDEX = os.path.join(_AUDIO_REFERS_DIR, "index.json")

MIN_SEC = 3.0
MAX_SEC = 10.0
JOIN_GAP = 1.2
# Below this the clip is unusable as a timbre reference at all; fall back to
# the longest line in the track rather than feeding the model 300ms of breath.
HARD_MIN_SEC = 1.0


def extract_refer_audio_main():
    demucs_audio()  # no-op when the stems already exist

    if not os.path.exists(VOCAL_HIFI_FILE):
        raise RuntimeError(f"missing separated vocal stem: {VOCAL_HIFI_FILE}")

    os.makedirs(_AUDIO_REFERS_DIR, exist_ok=True)
    df = pd.read_excel(_8_1_AUDIO_TASK)
    audio, sample_rate = sf.read(VOCAL_HIFI_FILE, always_2d=True)
    total_duration = len(audio) / sample_rate

    spans = [
        (parse_time(row["start_time"]), parse_time(row["end_time"]))
        for _, row in df.iterrows()
    ]
    fallback = pick_fallback_index(spans, min_sec=MIN_SEC)

    entries = {}
    expanded_count = 0
    fallback_count = 0

    for position, (_, row) in enumerate(df.iterrows()):
        number = int(row["number"])
        window = expand_window(
            spans,
            position,
            min_sec=MIN_SEC,
            max_sec=MAX_SEC,
            join_gap=JOIN_GAP,
            total_duration=total_duration,
        )
        used_fallback = False
        if window.duration < HARD_MIN_SEC and fallback is not None:
            window = expand_window(
                spans, fallback, min_sec=MIN_SEC, max_sec=MAX_SEC,
                join_gap=JOIN_GAP, total_duration=total_duration,
            )
            used_fallback = True
            fallback_count += 1
        elif window.duration > spans[position][1] - spans[position][0] + 1e-6:
            expanded_count += 1

        clip = _slice(audio, sample_rate, window.start, window.end)
        out_path = os.path.join(_AUDIO_REFERS_DIR, f"{number}.wav")
        sf.write(out_path, clip, sample_rate, subtype="PCM_16")

        # Pan is measured on the line's OWN window, never the fallback's: we
        # want where this line was spoken, not where the reference came from.
        pan = _measure_pan(audio, sample_rate, spans[position][0], spans[position][1])
        entries[str(number)] = {
            "number": number,
            "ref_start": round(window.start, 3),
            "ref_end": round(window.end, 3),
            "ref_duration": round(window.duration, 3),
            "line_start": round(spans[position][0], 3),
            "line_end": round(spans[position][1], 3),
            "pan": round(pan, 4),
            "fallback": used_fallback,
        }

    with open(REFER_INDEX, "w", encoding="utf-8") as fh:
        json.dump(
            {"sample_rate": sample_rate, "duration": total_duration, "lines": entries},
            fh,
            ensure_ascii=False,
            indent=2,
        )

    rprint(
        Panel(
            f"参考音频已写入 {_AUDIO_REFERS_DIR}\n"
            f"共 {len(entries)} 句，其中 {expanded_count} 句扩窗，"
            f"{fallback_count} 句过短已回退到全片最长句",
            title="Reference audio",
            border_style="green",
        )
    )


def _slice(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """Sample-accurate slice, clamped to the array and never empty."""
    first = max(0, int(round(start * sample_rate)))
    last = min(len(audio), int(round(end * sample_rate)))
    if last <= first:
        last = min(len(audio), first + int(0.2 * sample_rate))
    return audio[first:last]


def _measure_pan(audio: np.ndarray, sample_rate: int, start: float, end: float) -> float:
    """Stereo position of this line, from per-channel energy.

    Mono input is centre by definition. Energies are float64 sums; a 10s
    44.1kHz window cannot overflow.
    """
    if audio.shape[1] < 2:
        return 0.0
    clip = _slice(audio, sample_rate, start, end)
    if len(clip) == 0:
        return 0.0
    left = float(np.sum(np.square(clip[:, 0], dtype=np.float64)))
    right = float(np.sum(np.square(clip[:, 1], dtype=np.float64)))
    return pan_from_energy(left, right)


def load_refer_index() -> dict:
    """Read refers/index.json, or {} when the stage has not run."""
    if not os.path.exists(REFER_INDEX):
        return {}
    with open(REFER_INDEX, "r", encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    extract_refer_audio_main()
