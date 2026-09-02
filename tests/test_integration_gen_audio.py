"""Test the TTS stage against a fake GPU worker.

`core/_10_gen_audio.gen_audio` is the most intricate piece of upstream logic we
depend on: it calls our TTS backend per line, measures the real durations,
solves a speed factor per chunk, runs ffmpeg atempo, and writes the
`new_sub_times` that the mixdown consumes. It is also where a mistake in our
`custom_tts` contract surfaces -- and only after every line has been
synthesised for real, which on a T4 is 20-45 minutes.

So: a stdlib HTTP server that answers `/tts` with a sine wave whose length
tracks the requested text, and then the real stage on top of it. Everything
between our client and the mixdown gets exercised with no GPU and no model.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VL_ROOT = os.environ.get("ASMRDUB_VL_ROOT", "")

pytestmark = pytest.mark.skipif(
    not VL_ROOT or not os.path.isdir(VL_ROOT),
    reason="set ASMRDUB_VL_ROOT to a patched VideoLingo checkout",
)

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sf = pytest.importorskip("soundfile")

TTS_SR = 22050          # what IndexTTS emits
SECONDS_PER_CHAR = 0.22  # roughly Mandarin speaking rate

LINES = [
    # (start, end, chinese) -- line 3 is deliberately far too long for its slot,
    # which is what forces the speed-factor path.
    (1.0, 4.0, "晚上好，今天过得怎么样"),
    (4.5, 7.0, "我在这里陪着你"),
    (7.2, 8.4, "这句话特别长，长到无论怎么算都塞不进原来的时间轴里面去"),
    (12.0, 15.0, "那么，晚安"),
]


class _FakeTTS(BaseHTTPRequestHandler):
    """Answers /tts with a sine wave proportional to the text length."""

    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(payload)

        duration = max(0.3, len(payload["text"]) * SECONDS_PER_CHAR)
        factor = float(payload.get("duration_factor") or 1.0)
        duration *= factor
        samples = int(duration * TTS_SR)
        tone = (
            0.4 * np.sin(2 * np.pi * 200 * np.arange(samples) / TTS_SR)
        ).astype(np.float32)
        os.makedirs(os.path.dirname(payload["out"]) or ".", exist_ok=True)
        sf.write(payload["out"], tone, TTS_SR, subtype="PCM_16")

        body = json.dumps(
            {"ok": True, "out": payload["out"], "seconds": 0.01,
             "bytes": os.path.getsize(payload["out"])}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        body = json.dumps({"ok": True, "loaded": "tts"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Quiet: the pipeline's own output is what matters here."""


@pytest.fixture(scope="module")
def fake_worker():
    _FakeTTS.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeTTS)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", _FakeTTS
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """A checkout copy with a task table ready for the TTS stage."""
    base = tmp_path_factory.mktemp("gen")
    target = base / "VideoLingo"
    shutil.copytree(VL_ROOT, target, ignore=shutil.ignore_patterns("__pycache__", ".git"))

    sys.path.insert(0, ROOT)
    from asmrdub.srt_time import format_df

    audio_dir = target / "output" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # A raw mix long enough that the last line's trailing gap is positive:
    # analyze_subtitle_timing_and_speed derives it from the whole duration.
    total = 20.0
    sf.write(
        str(audio_dir / "raw.mp3"),
        np.zeros(int(total * 16000), dtype=np.float32),
        16000,
    )

    rows = [
        {
            "number": index,
            "start_time": format_df(start),
            "end_time": format_df(end),
            "duration": round(end - start, 3),
            "text": text,
            "origin": f"日本語 {index}",
        }
        for index, (start, end, text) in enumerate(LINES, start=1)
    ]
    pd.DataFrame(rows).to_excel(str(audio_dir / "tts_tasks.xlsx"), index=False)

    # Reference clips: 44.1kHz stereo, as _9_refer_audio would write them.
    refers = audio_dir / "refers"
    refers.mkdir(exist_ok=True)
    for index in range(1, len(LINES) + 1):
        clip = np.tile(
            (0.3 * np.sin(2 * np.pi * 180 * np.arange(int(3 * 44100)) / 44100)).astype(
                np.float32
            )[:, None],
            (1, 2),
        )
        sf.write(str(refers / f"{index}.wav"), clip, 44100, subtype="PCM_16")
    with open(refers / "index.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "sample_rate": 44100,
                "duration": total,
                "lines": {
                    str(i): {"number": i, "pan": 0.0, "fallback": False,
                             "ref_start": 0.0, "ref_end": 3.0, "ref_duration": 3.0,
                             "line_start": LINES[i - 1][0], "line_end": LINES[i - 1][1]}
                    for i in range(1, len(LINES) + 1)
                },
            },
            fh,
        )
    return target


def run(repo, worker_url: str, code: str) -> str:
    env = {
        **os.environ,
        "ASMRDUB_PKG_PATH": ROOT,
        "ASMRDUB_WORKER": worker_url,
        "PYTHONPATH": os.pathsep.join([ROOT, str(repo)]),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        pytest.fail(
            f"subprocess failed (rc={result.returncode})\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}"
        )
    return result.stdout


@pytest.fixture(scope="module")
def generated(repo, fake_worker):
    url, handler = fake_worker
    run(
        repo,
        url,
        "from core._8_2_dub_chunks import gen_dub_chunks; gen_dub_chunks()\n"
        "from core._10_gen_audio import gen_audio; gen_audio()\n",
    )
    return repo, handler


def test_every_line_was_synthesised(generated):
    repo, handler = generated
    assert len(handler.requests) >= len(LINES)
    texts = {request["text"] for request in handler.requests}
    for _, _, text in LINES:
        assert text in texts


def test_client_sends_the_right_reference_clip(generated):
    """Line N must clone from refers/N.wav -- this is what makes the dub
    follow the original speaker without diarization."""
    _, handler = generated
    by_text = {request["text"]: request for request in handler.requests}
    for index, (_, _, text) in enumerate(LINES, start=1):
        reference = by_text[text]["ref_audio"]
        assert os.path.basename(reference) == f"{index}.wav", (
            f"line {index} cloned from {reference}"
        )


def test_client_requests_chinese_and_no_padding(generated):
    _, handler = generated
    for request in handler.requests:
        assert request["lang"] == "ZH"
        # Padding inside a clip would shift its own end on an absolute timeline.
        assert request["interval_silence"] == 0


def test_no_emotion_vector_is_sent(generated):
    """Sending one would switch off reference-audio emotion transfer."""
    _, handler = generated
    for request in handler.requests:
        assert "emo_vector" not in request
        assert "use_emo_text" not in request


def test_duration_factor_only_slows_and_only_mildly(generated):
    _, handler = generated
    factors = [
        float(request["duration_factor"])
        for request in handler.requests
        if request.get("duration_factor")
    ]
    for factor in factors:
        assert 1.0 < factor <= 1.15, factor


def test_new_sub_times_are_written_for_every_row(generated):
    repo, _ = generated
    df = pd.read_excel(str(repo / "output" / "audio" / "tts_tasks.xlsx"))
    assert "new_sub_times" in df.columns
    for _, row in df.iterrows():
        times = ast.literal_eval(str(row["new_sub_times"]))
        assert times, f"row {row['number']} has no times"
        for start, end in times:
            assert end > start


def test_timeline_is_monotonic_and_non_overlapping(generated):
    """Overlapping windows would mean two dubs talking over each other."""
    repo, _ = generated
    df = pd.read_excel(str(repo / "output" / "audio" / "tts_tasks.xlsx"))
    flat = []
    for _, row in df.iterrows():
        flat.extend(ast.literal_eval(str(row["new_sub_times"])))
    flat.sort(key=lambda span: span[0])
    for (_, previous_end), (next_start, _) in zip(flat, flat[1:]):
        assert next_start >= previous_end - 1e-6, (
            f"overlap: previous ends {previous_end}, next starts {next_start}"
        )


def test_speed_factor_was_applied_to_the_overlong_line(generated):
    """Line 3 cannot fit its slot, so its clip must come out shortened."""
    repo, handler = generated
    long_text = LINES[2][2]
    raw = max(
        (r for r in handler.requests if r["text"] == long_text),
        key=lambda r: r.get("bytes", 0),
    )
    temp_path = raw["out"]
    final_path = os.path.join(
        str(repo), "output", "audio", "segs",
        os.path.basename(temp_path).replace("_temp", ""),
    )
    assert os.path.exists(final_path), f"missing {final_path}"
    original = sf.info(temp_path).duration
    adjusted = sf.info(final_path).duration
    assert adjusted < original * 0.99, (
        f"expected speed-up: {original:.2f}s -> {adjusted:.2f}s"
    )


def test_short_lines_are_left_at_natural_speed(generated):
    """Only lines that overrun get stretched; the rest must be untouched."""
    repo, handler = generated
    by_text = {request["text"]: request for request in handler.requests}
    temp_path = by_text[LINES[3][2]]["out"]
    final_path = os.path.join(
        str(repo), "output", "audio", "segs",
        os.path.basename(temp_path).replace("_temp", ""),
    )
    original = sf.info(temp_path).duration
    adjusted = sf.info(final_path).duration
    assert adjusted == pytest.approx(original, rel=0.02)


def test_segments_keep_the_tts_sample_rate(generated):
    """Resampling is the mixdown's job; the stage must not silently change it."""
    repo, _ = generated
    segs = os.path.join(str(repo), "output", "audio", "segs")
    for name in os.listdir(segs):
        if name.endswith(".wav"):
            assert sf.info(os.path.join(segs, name)).samplerate == TTS_SR


def test_mixdown_consumes_the_generated_stage(generated, fake_worker):
    """The real end-to-end seam: TTS output straight into the real mixdown."""
    repo, _ = generated
    url, _ = fake_worker
    audio_dir = repo / "output" / "audio"
    total = 20.0
    silence = np.zeros((int(total * 44100), 2), dtype=np.float32)
    rng = np.random.default_rng(7)
    bed = silence + rng.normal(0, 0.01, silence.shape).astype(np.float32)
    sf.write(str(audio_dir / "background_hifi.wav"), bed, 44100, subtype="PCM_16")
    sf.write(str(audio_dir / "vocal_hifi.wav"), silence, 44100, subtype="PCM_16")

    run(repo, url,
        "from core._11_merge_audio import merge_full_audio; merge_full_audio()")

    output = repo / "output" / "dub_44k_stereo.wav"
    assert output.exists()
    info = sf.info(str(output))
    assert info.samplerate == 44100 and info.channels == 2
    audio, _ = sf.read(str(output), dtype="float32")
    assert float(np.abs(audio).max()) <= 1.0
    with open(repo / "output" / "mix_report.json", encoding="utf-8") as fh:
        report = json.load(fh)
    assert report["segments"] >= len(LINES)
