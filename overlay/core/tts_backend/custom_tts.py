"""IndexTTS-2.5 backend for VideoLingo's `custom_tts` hook.

Contract (after the tts_main patch): custom_tts(text, save_path, number,
task_df). VideoLingo has already decided *what* to say and *how long* it may
take; this module only synthesises one line.

Every line is cloned from that line's own reference clip
(`output/audio/refers/{number}.wav`), which is why the dub follows the original
speaker without any diarization: whoever spoke line 42 in Japanese is whose
voice says line 42 in Chinese.

No emotion vector is passed. IndexTTS treats the speaker prompt as the emotion
prompt when `emo_vector` is absent, so the original performance's breath and
tremor transfer directly. Supplying a vector would switch that off.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.asr_backend.worker_client import TIMEOUT_TTS, call
from core.utils import load_key, rprint

REFERS_DIR = "output/audio/refers"

# IndexTTS emits 22.05kHz; VideoLingo's ffmpeg atempo pass and our mixdown both
# handle resampling, so nothing here needs to match the final rate.


def _reference_for(number: int) -> str:
    """Reference clip path for a line, building the clips if they are missing.

    Upstream orders the pipeline so `_9_refer_audio` runs before `_10_gen_audio`,
    but the UI lets a user re-run stages individually; regenerating is cheap
    compared to failing 200 lines in.
    """
    path = os.path.join(REFERS_DIR, f"{number}.wav")
    if os.path.exists(path):
        return path
    rprint(f"[yellow]⚠️ Reference clip {path} missing, extracting now ...[/yellow]")
    from core._9_refer_audio import extract_refer_audio_main

    extract_refer_audio_main()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"reference clip for line {number} still missing after extraction"
        )
    return path


def _duration_factor(number: int, task_df) -> float | None:
    """Ask IndexTTS to slow down when the line has room to breathe.

    VideoLingo's own mechanism is the opposite direction: it speeds audio up
    with ffmpeg atempo when the dub overruns. It has no way to use *spare*
    time, so a terse Chinese rendering of a long Japanese line ends up clipped
    and unnaturally brisk. IndexTTS's duration_factor fills that space inside
    the model, which sounds far better than stretching afterwards.

    Only mild slowdowns are requested (<= 1.15): the estimator is a syllable
    heuristic, and over-trusting it produces a drawl.
    """
    if task_df is None:
        return None
    try:
        row = task_df.loc[task_df["number"] == number]
        if row.empty:
            return None
        available = float(row.iloc[0].get("tol_dur") or 0.0)
        estimated = float(row.iloc[0].get("est_dur") or 0.0)
    except Exception:  # noqa: BLE001 - the column set varies by stage
        return None
    if available <= 0 or estimated <= 0:
        return None
    headroom = available / estimated
    if headroom <= 1.15:
        return None
    return min(1.15, round(headroom * 0.85, 3))


def custom_tts(text: str, save_path: str, number=None, task_df=None) -> None:
    """Synthesise one line. Signature matches upstream's patched call site.

    `number` is optional so a bare `custom_tts(text, path)` (upstream's
    unpatched signature, or a manual test) still works, falling back to the
    first reference clip.
    """
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if number is None:
        candidates = sorted(
            (p for p in os.listdir(REFERS_DIR) if p.endswith(".wav")),
            key=lambda name: int(name.split(".")[0]) if name[0].isdigit() else 1 << 30,
        ) if os.path.isdir(REFERS_DIR) else []
        if not candidates:
            raise FileNotFoundError(
                "custom_tts called without a line number and no reference clips exist"
            )
        reference = os.path.join(REFERS_DIR, candidates[0])
    else:
        reference = _reference_for(int(number))

    payload = {
        "text": text,
        "ref_audio": os.path.abspath(reference),
        "out": os.path.abspath(save_path),
        "lang": "ZH",
        # 0 rather than upstream's 200ms: this pipeline places every line on an
        # absolute timeline, so padding inside a clip would shift its own end.
        "interval_silence": 0,
        "max_tokens_per_segment": int(load_key("gui_seg_tokens"))
        if _has_key("gui_seg_tokens")
        else 120,
    }
    factor = _duration_factor(int(number), task_df) if number is not None else None
    if factor:
        payload["duration_factor"] = factor

    reply = call("/tts", payload, timeout=TIMEOUT_TTS)
    rprint(
        "[green]🔊 line %s -> %s (%.1fs gpu)[/green]"
        % (number, os.path.basename(save_path), reply.get("seconds", 0.0))
    )


def _has_key(key: str) -> bool:
    try:
        load_key(key)
        return True
    except Exception:  # noqa: BLE001 - load_key raises KeyError for absent keys
        return False


if __name__ == "__main__":
    custom_tts("这是一句测试。", "custom_tts_test.wav")
