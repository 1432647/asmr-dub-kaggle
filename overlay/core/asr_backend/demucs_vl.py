"""Replacement for VideoLingo's core/asr_backend/demucs_vl.py.

Upstream imports `demucs`, builds a Separator and runs it in-process on
`output/audio/raw.mp3` -- a 16kHz mono 32kbps file. Two problems for ASMR:

1. It needs torch in the app environment, which is the whole conflict we are
   avoiding (VideoLingo and IndexTTS cannot share a virtualenv).
2. Separating a 16k mono downmix throws away the binaural staging that is the
   entire point of the source material.

So this module keeps upstream's function name and contract -- other modules
call `demucs_audio()` and then read `_VOCAL_AUDIO_FILE` -- but does the work at
44.1kHz stereo in the GPU worker and additionally leaves the full-rate stems on
disk for reference-clip extraction and the final mixdown.
"""

from __future__ import annotations

import os
import subprocess

from core.asr_backend.worker_client import TIMEOUT_SEP, call
from core.utils import rprint
from core.utils.models import (
    _AUDIO_DIR,
    _BACKGROUND_AUDIO_FILE,
    _RAW_AUDIO_FILE,
    _VOCAL_AUDIO_FILE,
)

# Full-rate stereo stems: the real product of this stage.
VOCAL_HIFI_FILE = os.path.join(_AUDIO_DIR, "vocal_hifi.wav")
BACKGROUND_HIFI_FILE = os.path.join(_AUDIO_DIR, "background_hifi.wav")
# The original at full rate, kept so pan measurement and mixdown never have to
# re-decode the user's arbitrary input format.
SOURCE_HIFI_FILE = os.path.join(_AUDIO_DIR, "source_hifi.wav")

HIFI_SR = 44100


def _ffmpeg(args: list[str], what: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({what}): {result.stderr[-800:]}")


def prepare_source_hifi(media_file: str) -> str:
    """Decode the user's input to 44.1kHz stereo WAV once.

    Mono sources are duplicated to two channels so every downstream stage can
    assume stereo and skip channel-count branching.
    """
    if os.path.exists(SOURCE_HIFI_FILE):
        return SOURCE_HIFI_FILE
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    rprint(f"[blue]🎧 Decoding source to {HIFI_SR}Hz stereo ...[/blue]")
    _ffmpeg(
        ["-i", media_file, "-vn", "-ac", "2", "-ar", str(HIFI_SR),
         "-c:a", "pcm_s16le", SOURCE_HIFI_FILE],
        "source decode",
    )
    return SOURCE_HIFI_FILE


def demucs_audio() -> None:
    """Separate vocals from everything else, at full rate.

    Keeps upstream's name and its skip-if-done behaviour, so
    `_9_refer_audio.extract_refer_audio_main()` and anything else that calls it
    defensively still works.

    Also writes the 16k mono `vocal.mp3` / `background.mp3` that upstream code
    expects to exist, derived from the full-rate stems rather than separated
    separately.
    """
    if os.path.exists(VOCAL_HIFI_FILE) and os.path.exists(BACKGROUND_HIFI_FILE):
        rprint("[yellow]⚠️ Hi-fi stems already exist, skipping separation.[/yellow]")
        _ensure_legacy_stems()
        return

    if not os.path.exists(SOURCE_HIFI_FILE):
        raise RuntimeError(
            f"{SOURCE_HIFI_FILE} missing -- prepare_source_hifi() must run first"
        )

    rprint("[cyan]🎼 Separating vocals on the GPU worker (44.1kHz stereo) ...[/cyan]")
    reply = call(
        "/separate",
        {
            "input": os.path.abspath(SOURCE_HIFI_FILE),
            "vocal_out": os.path.abspath(VOCAL_HIFI_FILE),
            "background_out": os.path.abspath(BACKGROUND_HIFI_FILE),
            "sample_rate": HIFI_SR,
            "chunk_sec": 10.0,
            "overlap": 0.1,
        },
        timeout=TIMEOUT_SEP,
    )
    rprint(
        "[green]✨ Separated %.1fs in %d chunks[/green]"
        % (reply.get("duration", 0.0), reply.get("chunks", 0))
    )
    _ensure_legacy_stems()


def _ensure_legacy_stems() -> None:
    """Derive the 16k mono stems upstream names in core/utils/models.py.

    Cheap insurance: several upstream modules reference these paths, and an
    unexpected code path finding them missing would fail far from here.
    """
    for source, target in (
        (VOCAL_HIFI_FILE, _VOCAL_AUDIO_FILE),
        (BACKGROUND_HIFI_FILE, _BACKGROUND_AUDIO_FILE),
    ):
        if os.path.exists(target) or not os.path.exists(source):
            continue
        _ffmpeg(
            ["-i", source, "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
             "-b:a", "64k", target],
            f"downmix {os.path.basename(target)}",
        )


def raw_audio_exists() -> bool:
    return os.path.exists(_RAW_AUDIO_FILE)


if __name__ == "__main__":
    demucs_audio()
