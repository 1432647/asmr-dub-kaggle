"""Replacement for VideoLingo's core/_11_merge_audio.py and _12_dub_to_vid.py.

Upstream's merge is built for talking-head video: it downmixes every clip to
16kHz mono 64kbps mp3, concatenates them with silence, and leaves the ambience
bed to the video muxer -- which `_12_dub_to_vid` skips entirely for audio-only
input. Applied to binaural ASMR that produces a dry, mono, telephone-grade file
with no room and no staging.

This replacement:

* places each clip at an absolute sample offset on a 44.1kHz stereo canvas
  (concatenation drifts: every rounding error accumulates into the next gap);
* pans each dub to where the original voice sat, measured during reference
  extraction;
* keeps the separated ambience bed underneath at full rate;
* scales linearly if the sum would clip, rather than compressing -- ASMR
  dynamics are the product, not a mastering problem.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf
from rich.panel import Panel

from core._9_refer_audio import load_refer_index
from core.asr_backend.demucs_vl import BACKGROUND_HIFI_FILE
from core.utils import rprint
from core.utils.models import _8_1_AUDIO_TASK, _AUDIO_SEGS_DIR, _OUTPUT_DIR

sys.path.insert(0, os.environ.get("ASMRDUB_PKG_PATH", ""))
from asmrdub.mixdown import (  # noqa: E402
    canvas_length,
    equal_power_gains,
    limit_pan,
    peak_normalize_gain,
)
from asmrdub.srt_time import build_srt  # noqa: E402

MIX_SR = 44100
DUB_WAV = os.path.join(_OUTPUT_DIR, "dub_44k_stereo.wav")
DUB_MP3 = os.path.join(_OUTPUT_DIR, "dub.mp3")
DUB_SRT = os.path.join(_OUTPUT_DIR, "dub.srt")
MIX_REPORT = os.path.join(_OUTPUT_DIR, "mix_report.json")

# How much of the original ambience to keep under the dub. The bed still holds
# residual vocal energy after separation; at unity it fights the dub for the
# same frequencies, so it sits slightly back.
BACKGROUND_GAIN = 0.85
# Keep some centre bias: the pan estimate comes from a de-mixed stem and is
# noisy, and an over-panned dub is much more audible than an under-panned one.
MAX_SPREAD = 0.9
TAIL_SEC = 1.0


def merge_full_audio() -> None:
    df = pd.read_excel(_8_1_AUDIO_TASK)
    refer_index = load_refer_index().get("lines", {})

    segments = _collect_segments(df, refer_index)
    if not segments:
        raise RuntimeError(
            "no synthesised clips found -- run the TTS stage before merging"
        )

    background = _load_background()
    canvas_samples = canvas_length(
        [s["placement"] for s in segments],
        {(s["number"], s["line_index"]): s["length"] for s in segments},
        background_samples=len(background),
        tail_samples=int(TAIL_SEC * MIX_SR),
    )

    canvas = np.zeros((canvas_samples, 2), dtype=np.float32)
    if len(background):
        canvas[: len(background)] += background * BACKGROUND_GAIN

    for item in segments:
        placement = item["placement"]
        mono = item["audio"]
        start = placement.start_sample
        end = start + len(mono)
        canvas[start:end, 0] += mono * placement.gain_left
        canvas[start:end, 1] += mono * placement.gain_right

    peak = float(np.abs(canvas).max()) if canvas.size else 0.0
    gain = peak_normalize_gain(peak)
    if gain < 1.0:
        rprint(f"[yellow]⚠️ 峰值 {peak:.2f} 超过满刻度，整体衰减 {gain:.3f}[/yellow]")
        canvas *= gain

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    sf.write(DUB_WAV, canvas, MIX_SR, subtype="PCM_16")
    _encode_mp3(DUB_WAV, DUB_MP3)
    _write_srt(segments)

    with open(MIX_REPORT, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "sample_rate": MIX_SR,
                "duration": canvas_samples / MIX_SR,
                "segments": len(segments),
                "peak_before_gain": round(peak, 4),
                "applied_gain": round(gain, 4),
                "background_gain": BACKGROUND_GAIN,
                "background_present": bool(len(background)),
                "panned_segments": sum(
                    1 for s in segments if abs(s["placement"].pan) > 0.05
                ),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    rprint(
        Panel(
            f"成品：{DUB_WAV}\n"
            f"MP3：{DUB_MP3}\n"
            f"字幕：{DUB_SRT}\n"
            f"时长 {canvas_samples / MIX_SR:.1f}s · {len(segments)} 段配音 · "
            f"{'含环境音' if len(background) else '无环境音'}",
            title="混音完成",
            border_style="green",
        )
    )


def _collect_segments(df: pd.DataFrame, refer_index: dict) -> list[dict]:
    """Load every synthesised clip and compute its placement.

    Timing comes from `new_sub_times`, written by upstream's merge_chunks after
    it applied the speed factor -- those are the real post-stretch positions.
    """
    sys.path.insert(0, os.environ.get("ASMRDUB_PKG_PATH", ""))
    from asmrdub.mixdown import Placement

    segments = []
    for _, row in df.iterrows():
        number = int(row["number"])
        times = _as_list(row.get("new_sub_times"))
        lines = _as_list(row.get("lines"))
        if not times:
            rprint(f"[yellow]⚠️ 第 {number} 句缺少 new_sub_times，跳过[/yellow]")
            continue
        pan = limit_pan(
            float(refer_index.get(str(number), {}).get("pan", 0.0)), MAX_SPREAD
        )
        gains = equal_power_gains(pan)
        for line_index, span in enumerate(times):
            path = os.path.join(_AUDIO_SEGS_DIR, f"{number}_{line_index}.wav")
            if not os.path.exists(path):
                rprint(f"[yellow]⚠️ 缺少 {path}，跳过[/yellow]")
                continue
            mono = _load_mono_44k(path)
            start = float(span[0])
            segments.append(
                {
                    "number": number,
                    "line_index": line_index,
                    "audio": mono,
                    "length": len(mono),
                    "start": start,
                    "end": start + len(mono) / MIX_SR,
                    "text": lines[line_index] if line_index < len(lines) else "",
                    "placement": Placement(
                        number=number,
                        line_index=line_index,
                        start_sample=max(0, int(round(start * MIX_SR))),
                        pan=pan,
                        gain_left=gains.left,
                        gain_right=gains.right,
                    ),
                }
            )
    segments.sort(key=lambda item: (item["placement"].start_sample, item["number"]))
    return segments


def _as_list(value):
    """Parse a cell that pandas may hand back as a repr string.

    Upstream writes Python lists into xlsx cells, so they come back as strings
    like "[[1.0, 2.0]]". `eval` is upstream's own approach; ast.literal_eval is
    the same thing without arbitrary execution.
    """
    import ast

    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]


def _load_mono_44k(path: str) -> np.ndarray:
    """Read a clip as float32 mono at the mix rate.

    IndexTTS writes 22.05kHz mono; upstream's atempo pass preserves the rate.
    Resampling is delegated to ffmpeg rather than done here to avoid pulling in
    scipy/resampy for one operation.
    """
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate == MIX_SR:
        return np.ascontiguousarray(mono, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name
    try:
        _run_ffmpeg(
            ["-i", path, "-ac", "1", "-ar", str(MIX_SR), "-c:a", "pcm_f32le", temp_path],
            f"resample {os.path.basename(path)}",
        )
        resampled, _ = sf.read(temp_path, dtype="float32", always_2d=True)
        return np.ascontiguousarray(resampled[:, 0], dtype=np.float32)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _load_background() -> np.ndarray:
    """The separated ambience bed as (n, 2) float32, or an empty array."""
    if not os.path.exists(BACKGROUND_HIFI_FILE):
        rprint("[yellow]⚠️ 未找到环境音轨，只输出干声配音[/yellow]")
        return np.zeros((0, 2), dtype=np.float32)
    data, rate = sf.read(BACKGROUND_HIFI_FILE, dtype="float32", always_2d=True)
    if rate != MIX_SR:
        raise RuntimeError(f"background is {rate}Hz, expected {MIX_SR}Hz")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32)


def _write_srt(segments: list[dict]) -> None:
    entries = [(item["start"], item["end"], item["text"]) for item in segments]
    with open(DUB_SRT, "w", encoding="utf-8") as fh:
        fh.write(build_srt(entries))


def _encode_mp3(source: str, target: str) -> None:
    """192kbps stereo -- upstream's 64kbps mono would undo the whole point."""
    _run_ffmpeg(
        ["-i", source, "-c:a", "libmp3lame", "-b:a", "192k", "-ar", str(MIX_SR), target],
        "mp3 encode",
    )


def _run_ffmpeg(args: list[str], what: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({what}): {result.stderr[-800:]}")


def merge_video_audio() -> None:
    """Stand-in for `_12_dub_to_vid.merge_video_audio` -- audio-only pipeline.

    Kept so the stage list can stay parallel to upstream's without a branch.
    """
    rprint("[green]🎵 纯音频流程，跳过视频合成。[/green]")


if __name__ == "__main__":
    merge_full_audio()
