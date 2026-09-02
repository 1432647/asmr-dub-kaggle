"""Replacement for VideoLingo's core/_2_asr.py.

Differences from upstream, all forced:

* whisperX -> faster-whisper. whisperX pins torch~=2.8 plus pyannote and
  torchcodec, which cannot coexist with IndexTTS in one environment. ASR now
  happens in the GPU worker over HTTP.
* Separation runs first and at full rate, because the reference clips for voice
  cloning and the final ambience bed both come from those stems.
* No 30-minute audio splitting. Upstream splits long inputs at silence points
  because whisperX loads the whole file into VRAM; faster-whisper streams, and a
  15-minute ASMR track has no reliable 60-second-wide silence window anyway.

Kept from upstream: the `@check_file_exists(_2_CLEANED_CHUNKS)` guard, so a
re-run after a crash skips straight past this stage.
"""

from __future__ import annotations

import os

from core._1_ytdlp import find_media_file
from core.asr_backend.audio_preprocess import (
    convert_video_to_audio,
    prepare_audio_for_asr,
    process_transcription,
    save_results,
)
from core.asr_backend.demucs_vl import demucs_audio, prepare_source_hifi
from core.asr_backend.worker_client import TIMEOUT_ASR, call
from core.utils import check_file_exists, load_key, rprint, update_key
from core.utils.models import _2_CLEANED_CHUNKS, _RAW_AUDIO_FILE


@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    media_file, media_type = find_media_file()
    rprint(f"[cyan]📥 Input: {media_file} ({media_type})[/cyan]")

    # 1. 16k mono for ASR (upstream's own helpers).
    if media_type == "video":
        convert_video_to_audio(media_file)
    else:
        prepare_audio_for_asr(media_file)

    # 2. Full-rate stereo decode + separation. Must precede ASR: the worker
    #    holds one model at a time, and doing separation first means the
    #    reference clips are ready before anything needs them.
    prepare_source_hifi(media_file)
    demucs_audio()

    # 3. Transcribe. Feed the ORIGINAL 16k mix rather than the separated vocal:
    #    HDemucs is a music model and leaves artefacts on speech that whisper
    #    hears as extra syllables.
    language = load_key("whisper.language")
    rprint(f"[cyan]🎤 Transcribing with faster-whisper (language={language}) ...[/cyan]")
    reply = call(
        "/asr",
        {
            "input": os.path.abspath(_RAW_AUDIO_FILE),
            "language": None if language in ("auto", "", None) else language,
        },
        timeout=TIMEOUT_ASR,
    )
    result = reply["result"]

    detected = result.get("language") or language
    if detected:
        update_key("whisper.detected_language", detected)
    rprint(
        "[green]✅ %d segments, language=%s[/green]"
        % (len(result.get("segments", [])), detected)
    )
    if not result.get("segments"):
        raise RuntimeError(
            "ASR returned no speech. Check that the upload is not silent and "
            "that the VAD threshold is not too high for this recording."
        )

    # 4. Upstream's word-level flattening and xlsx writing, unchanged.
    df = process_transcription(result)
    save_results(df)


if __name__ == "__main__":
    transcribe()
