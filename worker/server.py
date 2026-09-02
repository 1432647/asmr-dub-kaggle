#!/usr/bin/env python3
"""P2 -- the GPU worker: ASR, source separation and TTS behind one HTTP port.

Why a separate process at all: VideoLingo pins librosa 0.11 / opencv 4.11 /
numpy>=2.0.2 while IndexTTS pins librosa 0.10.2 / opencv 4.9 /
transformers 4.52.1. They cannot share a virtualenv. So all torch work lives
here, in index-tts's own uv environment, and the app process (P1) stays
torch-free and talks to it over loopback HTTP.

Why one lock: a single T4 holds roughly one of these models at a time.
Requests are serialised and models are evicted when a different kind of work
arrives, which is safe because the pipeline stages are sequential anyway
(separate -> transcribe -> synthesise).

Only the stdlib is used for serving -- no fastapi/uvicorn -- so this adds
nothing to the dependency surface of an already fragile environment.

SECURITY: binds 127.0.0.1 only and has no authentication. It must never be
exposed through the tunnel; only P1 (same container) may reach it.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# Paths. IndexTTS hardcodes './checkpoints/hf_cache' as HF_HUB_CACHE at import
# time (a *relative* path), and reads auxiliary weights from
# {model_dir}/hf_cache/. We therefore chdir into the index-tts checkout and
# make ./checkpoints resolve to the real model directory before importing it.
# --------------------------------------------------------------------------

STATE = {
    "index_tts_root": None,
    "model_dir": None,
    "whisper_dir": None,
    "hdemucs_path": None,
    "device": "cuda:0",
    "loaded": None,      # "tts" | "asr" | "sep" | None
    "tts": None,
    "asr": None,
    "sep": None,
    "log": None,
}

LOCK = threading.Lock()


def log(message: str) -> None:
    line = "[worker %s] %s" % (time.strftime("%H:%M:%S"), message)
    print(line, flush=True)
    path = STATE.get("log")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


# --------------------------------------------------------------------------
# Model lifecycle
# --------------------------------------------------------------------------


def _free_vram() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_all() -> None:
    """Drop every model. Called before loading a different kind of work."""
    for key in ("tts", "asr", "sep"):
        if STATE.get(key) is not None:
            log("unloading %s" % key)
            STATE[key] = None
    STATE["loaded"] = None
    _free_vram()


def ensure_tts():
    if STATE["loaded"] == "tts" and STATE["tts"] is not None:
        return STATE["tts"]
    unload_all()
    log("loading IndexTTS-2.5 ...")
    os.chdir(STATE["index_tts_root"])
    if STATE["index_tts_root"] not in sys.path:
        sys.path.insert(0, STATE["index_tts_root"])
    from indextts.infer_v2_5 import IndexTTS2

    cfg_path = os.path.join(STATE["model_dir"], "config.yaml")
    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=STATE["model_dir"],
        # T4 is sm_75: no native bfloat16. torch.cuda.is_bf16_supported()
        # returns True via emulation, which is a trap -- it would be slower
        # than fp32 and can produce NaNs. Keep full precision.
        use_bf16=False,
        device=STATE["device"],
        # BigVGAN's fused CUDA kernel is JIT-compiled by ninja at load time and
        # regularly fails in sandboxes; the torch fallback is fast enough.
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        # Every line already has real reference audio, which IndexTTS uses as
        # its emotion source. Guessing emotion from Chinese text would be
        # strictly worse AND would switch the reference-audio emotion path off.
        use_qwen_emo=False,
    )
    STATE["tts"] = tts
    STATE["loaded"] = "tts"
    log("IndexTTS-2.5 ready")
    return tts


def ensure_asr():
    if STATE["loaded"] == "asr" and STATE["asr"] is not None:
        return STATE["asr"]
    unload_all()
    log("loading faster-whisper large-v3 ...")
    from faster_whisper import WhisperModel

    model = WhisperModel(
        STATE["whisper_dir"],
        device="cuda",
        device_index=int(STATE["device"].rsplit(":", 1)[-1]),
        # T4 has real fp16 tensor cores; int8_float16 would be faster but
        # measurably worse on whispered speech, which is already the weak link.
        compute_type="float16",
    )
    STATE["asr"] = model
    STATE["loaded"] = "asr"
    log("faster-whisper ready")
    return model


def ensure_sep():
    if STATE["loaded"] == "sep" and STATE["sep"] is not None:
        return STATE["sep"]
    unload_all()
    log("loading HDemucs ...")
    import torch
    from torchaudio.models import hdemucs_high

    model = hdemucs_high(sources=list(SOURCES))
    state_dict = torch.load(STATE["hdemucs_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(STATE["device"])
    STATE["sep"] = model
    STATE["loaded"] = "sep"
    log("HDemucs ready")
    return model


SOURCES = ("drums", "bass", "other", "vocals")


# --------------------------------------------------------------------------
# Source separation
# --------------------------------------------------------------------------


def do_separate(payload: dict) -> dict:
    """Split a file into vocals and everything-else at full rate.

    Runs at 44.1kHz stereo, unlike VideoLingo's own pass which separates a
    16kHz mono 32kbps mp3. For binaural ASMR that downmix destroys the entire
    point: the stereo movement of the voice IS the content.
    """
    import torch
    import torchaudio

    sys.path.insert(0, payload["asmrdub_path"])
    from asmrdub.chunking import fade_envelope, plan_chunks, verify_coverage

    src = payload["input"]
    vocal_out = payload["vocal_out"]
    background_out = payload["background_out"]
    chunk_sec = float(payload.get("chunk_sec", 10.0))
    overlap = float(payload.get("overlap", 0.1))
    sample_rate = int(payload.get("sample_rate", 44100))

    waveform, in_sr = torchaudio.load(src)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)      # HDemucs expects 2 channels
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    if in_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, in_sr, sample_rate)

    total = waveform.shape[1]
    chunks = plan_chunks(total, sample_rate, chunk_sec=chunk_sec, overlap=overlap)
    if not verify_coverage(chunks, total):
        raise RuntimeError("chunk plan does not cover the input")
    log("separating %.1fs in %d chunks" % (total / sample_rate, len(chunks)))

    # HDemucs was trained on standardised input; normalising by the whole
    # track (not per chunk) keeps the chunks mutually consistent.
    ref = waveform.mean(0)
    mean, std = ref.mean().item(), ref.std().item() or 1e-8
    normalised = (waveform - mean) / std

    model = ensure_sep()
    vocal_index = SOURCES.index("vocals")
    vocal = torch.zeros_like(waveform)
    background = torch.zeros_like(waveform)
    device = STATE["device"]

    with torch.no_grad():
        for chunk in chunks:
            piece = normalised[:, chunk.start:chunk.end].unsqueeze(0).to(device)
            estimated = model(piece)[0].cpu()      # (source, channel, time)
            # Undo the normalisation with std only. `mean` is a property of the
            # whole mixture, not of each stem: adding it to all four would add
            # it four times over when the stems are recombined. It goes back
            # onto the background alone, below, so vocal+background == input.
            estimated = estimated * std
            # Shared, unit-tested envelope: the seam gains sum to exactly 1, so
            # the crossfade region is neither louder nor quieter than the rest.
            envelope = torch.tensor(fade_envelope(chunk), dtype=estimated.dtype)
            voc = estimated[vocal_index] * envelope
            other = (estimated.sum(0) - estimated[vocal_index]) * envelope
            vocal[:, chunk.start:chunk.end] += voc
            background[:, chunk.start:chunk.end] += other
            del piece, estimated
            if chunk.index % 8 == 0:
                _free_vram()
                log("  chunk %d/%d" % (chunk.index + 1, len(chunks)))

    # The mixture's DC offset belongs to exactly one stem. Putting it on the
    # background keeps `vocal + background == input` sample-exact, and keeps the
    # reference clips (cut from the vocal stem) free of an offset that would
    # bias IndexTTS's speaker encoder.
    background = background + mean

    # Clip guard. A single shared gain, not one per stem: scaling them
    # independently would change their balance and break the
    # `vocal + background == input` property that the mixdown stage relies on.
    peak = max(vocal.abs().max().item(), background.abs().max().item())
    gain = 1.0 / peak if peak > 1.0 else 1.0
    if gain != 1.0:
        log("peak %.3f > 1.0, scaling both stems by %.4f" % (peak, gain))

    for path, tensor in ((vocal_out, vocal), (background_out, background)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torchaudio.save(
            path, tensor * gain, sample_rate, encoding="PCM_S", bits_per_sample=16
        )
        log("wrote %s" % path)

    return {
        "ok": True,
        "sample_rate": sample_rate,
        "channels": int(vocal.shape[0]),
        "duration": total / sample_rate,
        "chunks": len(chunks),
        "gain": gain,
    }


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------


def do_asr(payload: dict) -> dict:
    """Transcribe with word timestamps and return VideoLingo's segment shape."""
    sys.path.insert(0, payload["asmrdub_path"])
    from asmrdub.asr_format import to_videolingo

    model = ensure_asr()
    src = payload["input"]
    language = payload.get("language") or None
    log("transcribing %s (language=%s)" % (os.path.basename(src), language))

    segments, info = model.transcribe(
        src,
        language=language,
        task="transcribe",
        beam_size=int(payload.get("beam_size", 5)),
        word_timestamps=True,
        # Whispered ASMR sits close to the VAD floor; the default 0.5 threshold
        # drops entire lines. Lower it and allow long pauses between phrases.
        vad_filter=True,
        vad_parameters={
            "threshold": float(payload.get("vad_threshold", 0.25)),
            "min_speech_duration_ms": 120,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 200,
        },
        # Deterministic greedy first, temperature fallback only on failure --
        # random sampling on breathy audio invents text.
        temperature=[0.0, 0.2, 0.4],
        condition_on_previous_text=False,  # stops runaway repetition loops
        no_speech_threshold=0.7,
    )
    materialised = list(segments)   # the generator does the actual work
    detected = getattr(info, "language", None) or language or "ja"
    result = to_videolingo(materialised, detected)
    log("transcribed %d segments (language=%s)" % (len(result["segments"]), detected))
    return {"ok": True, "result": result}


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------


def do_tts(payload: dict) -> dict:
    """Clone one line from its own reference clip."""
    tts = ensure_tts()
    text = payload["text"]
    ref = payload["ref_audio"]
    out = payload["out"]
    if not os.path.isfile(ref):
        raise FileNotFoundError(f"reference audio missing: {ref}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    kwargs = dict(
        spk_audio_prompt=ref,
        text=text,
        output_path=out,
        lang=payload.get("lang", "ZH"),
        verbose=bool(payload.get("verbose", False)),
        max_text_tokens_per_segment=int(payload.get("max_tokens_per_segment", 120)),
        interval_silence=int(payload.get("interval_silence", 0)),
    )
    factor = payload.get("duration_factor")
    if factor:
        # >1 slows down, <1 speeds up; valid range 0.5-2.0 per the model card.
        kwargs["duration_factor"] = max(0.5, min(2.0, float(factor)))

    started = time.perf_counter()
    tts.infer(**kwargs)
    if not os.path.isfile(out):
        raise RuntimeError("IndexTTS reported success but produced no file")
    return {
        "ok": True,
        "out": out,
        "bytes": os.path.getsize(out),
        "seconds": round(time.perf_counter() - started, 2),
    }


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------

ROUTES = {
    "/separate": do_separate,
    "/asr": do_asr,
    "/tts": do_tts,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") in ("/health", ""):
            self._send(
                200,
                {
                    "ok": True,
                    "loaded": STATE["loaded"],
                    "device": STATE["device"],
                    "vram": _vram_report(),
                },
            )
        elif self.path.rstrip("/") == "/unload":
            with LOCK:
                unload_all()
            self._send(200, {"ok": True})
        else:
            self._send(404, {"ok": False, "error": "no such route"})

    def do_POST(self):  # noqa: N802
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        handler = ROUTES.get(route)
        if handler is None:
            self._send(404, {"ok": False, "error": f"no such route: {route}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as exc:
            self._send(400, {"ok": False, "error": f"bad json: {exc}"})
            return
        # One GPU, one job. Serialising here means a slow request blocks the
        # next one rather than both OOM-ing.
        with LOCK:
            try:
                self._send(200, handler(payload))
            except Exception as exc:  # noqa: BLE001 - report, never die
                detail = traceback.format_exc(limit=8)
                log("ERROR %s: %s" % (route, exc))
                log(detail)
                self._send(
                    500,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                     "traceback": detail},
                )

    def log_message(self, fmt, *args):
        """Silence per-request access logs; we log meaningful events instead."""


def _vram_report() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = int(STATE["device"].rsplit(":", 1)[-1])
        free, total = torch.cuda.mem_get_info(index)
        return {"free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2)}
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="ASMR dubbing GPU worker")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--index-tts-root", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="IndexTTS-2.5 checkpoints directory")
    parser.add_argument("--whisper-dir", required=True)
    parser.add_argument("--hdemucs", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--log")
    args = parser.parse_args()

    # Pin to one GPU *before* torch initialises, so the LLM on the other card
    # is invisible here and cuda:0 always means our card.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    STATE["index_tts_root"] = os.path.abspath(args.index_tts_root)
    STATE["model_dir"] = os.path.abspath(args.model_dir)
    STATE["whisper_dir"] = os.path.abspath(args.whisper_dir)
    STATE["hdemucs_path"] = os.path.abspath(args.hdemucs)
    STATE["device"] = "cuda:0"
    STATE["log"] = os.path.abspath(args.log) if args.log else None

    # IndexTTS resolves auxiliary weights relative to the cwd, so anchor there
    # and make ./checkpoints point at wherever the weights actually live.
    os.chdir(STATE["index_tts_root"])
    local_ckpt = os.path.join(STATE["index_tts_root"], "checkpoints")
    if os.path.abspath(local_ckpt) != STATE["model_dir"]:
        _link_or_note(local_ckpt, STATE["model_dir"])

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log("listening on http://%s:%d (gpu %s)" % (args.host, args.port, args.gpu))
    log("index-tts root: %s" % STATE["index_tts_root"])
    log("model dir:      %s" % STATE["model_dir"])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    return 0


def _link_or_note(link_path: str, target: str) -> None:
    """Point ./checkpoints at the real model dir, replacing a stale link."""
    try:
        if os.path.islink(link_path) or os.path.isfile(link_path):
            os.unlink(link_path)
        elif os.path.isdir(link_path):
            if os.listdir(link_path):
                log("WARNING %s exists and is not empty; leaving it alone" % link_path)
                return
            os.rmdir(link_path)
        os.symlink(target, link_path)
        log("linked %s -> %s" % (link_path, target))
    except OSError as exc:
        log("WARNING could not link checkpoints (%s); relying on absolute paths" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
