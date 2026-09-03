"""Offline end-to-end test of the audio pipeline, no GPU and no models.

Runs the real overlay modules -- `_8_2_dub_chunks`, `_9_refer_audio`,
`_11_merge_audio` -- against synthetic audio and a synthetic task table. This
is what catches the failures that only appear when the pieces are wired
together: a column upstream expects and we forgot to write, a timestamp format
mismatch, a mono/stereo confusion, a pan that lands on the wrong ear.

Requires a patched VideoLingo checkout; set ASMRDUB_VL_ROOT to it. Create one
with:

    python bootstrap/apply_overlay.py --repo-root <checkout> --overlay-root overlay

Skipped when that variable is unset, so `pytest` stays green on a bare clone.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys

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

SR = 44100
LINES = [
    # (start, end, japanese, chinese, pan) -- pan drives which ear gets it.
    # The durations and gaps are chosen to exercise every reference-window
    # branch: long enough already / short with a close neighbour / short and
    # isolated.
    (1.0, 4.2, "こんばんは、よく眠れましたか", "晚上好，睡得好吗", -0.8),      # 3.2s, no expansion
    (4.8, 5.5, "ふふっ", "呵呵", 0.8),                                    # 0.7s, borrows from next
    (6.0, 9.5, "今日はとても静かですね", "今天真安静呢", 0.8),               # 3.5s, no expansion
    (13.0, 15.5, "また明日ね、おやすみなさい", "明天见，晚安", -0.2),          # 2.5s, isolated
]
TOTAL_SEC = 18.0


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A VideoLingo checkout copy with synthetic audio and task table.

    The module is copied rather than used in place so a failing test cannot
    leave the caller's checkout dirty.
    """
    base = tmp_path_factory.mktemp("vl")
    repo = base / "VideoLingo"
    shutil.copytree(VL_ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".git"))

    audio_dir = repo / "output" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    _write_synthetic_stems(audio_dir)
    _write_task_table(repo)
    _write_manifest(repo)
    return repo


def _write_synthetic_stems(audio_dir) -> None:
    """A stereo 'vocal' stem panned per line, plus a steady ambience bed."""
    samples = int(TOTAL_SEC * SR)
    time = np.arange(samples) / SR
    vocal = np.zeros((samples, 2), dtype=np.float32)

    for index, (start, end, _, _, pan) in enumerate(LINES):
        first, last = int(start * SR), int(end * SR)
        tone = 0.35 * np.sin(2 * np.pi * (180 + 40 * index) * time[first:last])
        left = float(np.sqrt((1 - pan) / 2))
        right = float(np.sqrt((1 + pan) / 2))
        vocal[first:last, 0] += tone * left
        vocal[first:last, 1] += tone * right

    # Wide, quiet noise bed: what Demucs would hand back as "everything else".
    rng = np.random.default_rng(1234)
    background = (rng.normal(0, 0.02, (samples, 2))).astype(np.float32)

    sf.write(str(audio_dir / "vocal_hifi.wav"), vocal, SR, subtype="PCM_16")
    sf.write(str(audio_dir / "background_hifi.wav"), background, SR, subtype="PCM_16")
    sf.write(str(audio_dir / "source_hifi.wav"), vocal + background, SR, subtype="PCM_16")
    # The 16k mono mix upstream helpers reference.
    mono = (vocal + background).mean(axis=1)
    sf.write(str(audio_dir / "raw.mp3"), mono[:: SR // 16000], 16000)


def _write_task_table(repo) -> None:
    """A tts_tasks.xlsx shaped like _8_1_audio_task.gen_audio_task_main writes."""
    sys.path.insert(0, ROOT)
    from asmrdub.srt_time import format_df

    rows = []
    for number, (start, end, origin, text, _) in enumerate(LINES, start=1):
        rows.append(
            {
                "number": number,
                "start_time": format_df(start),
                "end_time": format_df(end),
                "duration": round(end - start, 3),
                "text": text,
                "origin": origin,
            }
        )
    target = repo / "output" / "audio" / "tts_tasks.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(target, index=False)


def _write_manifest(repo) -> None:
    manifest = repo / "output" / "input_manifest.json"
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(
            {"path": "output/audio/source_hifi.wav", "type": "audio"},
            fh,
        )


def run_in_repo(repo, code: str) -> str:
    """Execute ``code`` with the repo as cwd and the app-side env configured.

    A subprocess rather than an import: these modules mutate sys.path, chdir,
    and read config.yaml relative to the cwd, and VideoLingo's `load_key`
    caches nothing -- running them in-process would leak state between tests.
    """
    env = {
        **os.environ,
        "ASMRDUB_PKG_PATH": ROOT,
        "PYTHONPATH": os.pathsep.join([ROOT, str(repo)]),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        pytest.fail(
            f"subprocess failed (rc={result.returncode})\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}"
        )
    return result.stdout


# --------------------------------------------------------------------------
# Stage: dub chunks
# --------------------------------------------------------------------------


def test_dub_chunks_adds_timing_columns(workspace):
    run_in_repo(
        workspace,
        "from core._8_2_dub_chunks import gen_dub_chunks; gen_dub_chunks()",
    )
    df = pd.read_excel(workspace / "output" / "audio" / "tts_tasks.xlsx")
    for column in ("gap", "tolerance", "tol_dur", "est_dur", "if_too_fast",
                   "cut_off", "lines", "src_lines"):
        assert column in df.columns, f"missing {column}"
    assert len(df) == len(LINES)


def test_dub_chunks_maps_one_line_per_row(workspace):
    """The 1:1 replacement for upstream's fragile string matching."""
    df = pd.read_excel(workspace / "output" / "audio" / "tts_tasks.xlsx")
    import ast

    for _, row in df.iterrows():
        lines = ast.literal_eval(str(row["lines"]))
        assert isinstance(lines, list) and len(lines) == 1
        assert lines[0] == row["text"]


def test_dub_chunks_survives_edited_translations(workspace):
    """Upstream would raise 'Matching failed' here; that is the whole point."""
    path = workspace / "output" / "audio" / "tts_tasks.xlsx"
    df = pd.read_excel(path)
    df.loc[0, "text"] = "这句话被人工完全改写了，和 trans.srt 完全不匹配"
    df.to_excel(path, index=False)
    run_in_repo(
        workspace,
        "from core._8_2_dub_chunks import gen_dub_chunks; gen_dub_chunks()",
    )
    import ast

    after = pd.read_excel(path)
    assert "人工完全改写" in ast.literal_eval(str(after.loc[0, "lines"]))[0]


def test_dub_chunks_marks_at_least_one_cutoff(workspace):
    df = pd.read_excel(workspace / "output" / "audio" / "tts_tasks.xlsx")
    assert df["cut_off"].sum() >= 1
    assert df.iloc[-1]["cut_off"] == 1, "the last row must close its chunk"


# --------------------------------------------------------------------------
# Stage: reference audio
# --------------------------------------------------------------------------


def test_reference_clips_are_written_for_every_line(workspace):
    run_in_repo(
        workspace,
        "from core._9_refer_audio import extract_refer_audio_main;"
        "extract_refer_audio_main()",
    )
    refers = workspace / "output" / "audio" / "refers"
    for number in range(1, len(LINES) + 1):
        assert (refers / f"{number}.wav").exists()
    assert (refers / "index.json").exists()


def test_reference_clips_keep_full_rate_stereo(workspace):
    """Cloning from a 16k mono downmix would throw away breath and sibilance."""
    clip = workspace / "output" / "audio" / "refers" / "1.wav"
    info = sf.info(str(clip))
    assert info.samplerate == SR
    assert info.channels == 2


def test_short_line_reference_is_expanded(workspace):
    """Line 2 is 0.7s; it must borrow from line 3 across the 0.5s gap."""
    with open(
        workspace / "output" / "audio" / "refers" / "index.json", encoding="utf-8"
    ) as fh:
        index = json.load(fh)
    entry = index["lines"]["2"]
    assert entry["line_end"] - entry["line_start"] == pytest.approx(0.7, abs=0.01)
    assert entry["ref_duration"] > 2.0, entry
    duration = sf.info(
        str(workspace / "output" / "audio" / "refers" / "2.wav")
    ).duration
    assert duration == pytest.approx(entry["ref_duration"], abs=0.05)


def test_long_line_reference_is_not_expanded(workspace):
    """Line 1 is already 3.2s, over the 3.0s floor: leave it alone."""
    with open(
        workspace / "output" / "audio" / "refers" / "index.json", encoding="utf-8"
    ) as fh:
        index = json.load(fh)
    entry = index["lines"]["1"]
    assert entry["ref_duration"] == pytest.approx(3.2, abs=0.01)
    assert entry["ref_start"] == pytest.approx(entry["line_start"], abs=0.01)


def test_isolated_short_line_is_not_stitched_across_a_long_gap(workspace):
    """Line 4 sits after a 3.5s silence -- a likely speaker change."""
    with open(
        workspace / "output" / "audio" / "refers" / "index.json", encoding="utf-8"
    ) as fh:
        index = json.load(fh)
    entry = index["lines"]["4"]
    assert entry["ref_duration"] == pytest.approx(2.5, abs=0.01)
    assert entry["fallback"] is False


def test_measured_pan_reproduces_the_original_channel_balance(workspace):
    """Measure-then-place must reproduce the source's L/R ratio.

    The fixture pans with a linear-energy law, deliberately *not* the
    equal-power law the mixdown uses -- a real recording follows no particular
    law either. What has to hold is the round trip: whatever ratio was
    measured, re-placing a mono dub at that pan must land it with the same
    ratio. Asserting an exact pan number instead would only be testing one
    law against itself.
    """
    sys.path.insert(0, ROOT)
    from asmrdub.mixdown import equal_power_gains

    with open(
        workspace / "output" / "audio" / "refers" / "index.json", encoding="utf-8"
    ) as fh:
        index = json.load(fh)

    for number, (_, _, _, _, pan) in enumerate(LINES, start=1):
        measured = index["lines"][str(number)]["pan"]
        source_ratio = math.sqrt((1 + pan) / 2) / max(1e-9, math.sqrt((1 - pan) / 2))
        gains = equal_power_gains(measured)
        placed_ratio = gains.right / max(1e-9, gains.left)
        assert placed_ratio == pytest.approx(source_ratio, rel=0.05), (
            f"line {number}: source R/L {source_ratio:.3f}, "
            f"replaced at pan {measured} gives {placed_ratio:.3f}"
        )


def test_measured_pan_keeps_sign_and_order(workspace):
    """A left-side voice must never be measured as right-side."""
    with open(
        workspace / "output" / "audio" / "refers" / "index.json", encoding="utf-8"
    ) as fh:
        index = json.load(fh)
    measured = [index["lines"][str(n)]["pan"] for n in range(1, len(LINES) + 1)]
    intended = [line[4] for line in LINES]
    for value, want in zip(measured, intended):
        if abs(want) > 0.1:
            assert (value > 0) == (want > 0), f"pan sign flipped: {value} vs {want}"
    order_measured = sorted(range(len(measured)), key=lambda i: measured[i])
    order_intended = sorted(range(len(intended)), key=lambda i: intended[i])
    assert order_measured == order_intended


# --------------------------------------------------------------------------
# Stage: mixdown
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed(workspace):
    """Fake the TTS stage, then run the real mixdown."""
    segs = workspace / "output" / "audio" / "segs"
    segs.mkdir(parents=True, exist_ok=True)

    path = workspace / "output" / "audio" / "tts_tasks.xlsx"
    df = pd.read_excel(path)
    sys.path.insert(0, ROOT)
    from asmrdub.srt_time import parse_time

    times = []
    for _, row in df.iterrows():
        number = int(row["number"])
        start = parse_time(row["start_time"])
        length = max(0.4, parse_time(row["end_time"]) - start - 0.1)
        # 22.05kHz mono, matching what IndexTTS actually emits.
        tone = 0.5 * np.sin(
            2 * np.pi * 220 * np.arange(int(length * 22050)) / 22050
        ).astype(np.float32)
        sf.write(str(segs / f"{number}_0.wav"), tone, 22050, subtype="PCM_16")
        times.append([[start, start + length]])
    df["new_sub_times"] = [str(t) for t in times]
    df.to_excel(path, index=False)

    run_in_repo(
        workspace,
        "from core._11_merge_audio import merge_full_audio; merge_full_audio()",
    )
    return workspace / "output"


def test_mixdown_writes_all_outputs(mixed):
    assert (mixed / "dub_44k_stereo.wav").exists()
    assert (mixed / "dub.mp3").exists()
    assert (mixed / "dub.srt").exists()
    assert (mixed / "mix_report.json").exists()


def test_mixdown_is_full_rate_stereo(mixed):
    info = sf.info(str(mixed / "dub_44k_stereo.wav"))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.duration >= TOTAL_SEC


def test_mixdown_does_not_clip(mixed):
    audio, _ = sf.read(str(mixed / "dub_44k_stereo.wav"), dtype="float32")
    assert float(np.abs(audio).max()) <= 1.0


def test_dub_lands_on_the_same_ear_as_the_original(mixed):
    """Line 1 is hard left in the source; its dub must be louder on the left."""
    audio, rate = sf.read(str(mixed / "dub_44k_stereo.wav"), dtype="float32")
    for start, end, _, _, pan in LINES:
        if abs(pan) < 0.5:
            continue
        window = audio[int((start + 0.1) * rate) : int((end - 0.1) * rate)]
        if len(window) < rate // 10:
            continue
        left = float(np.sum(window[:, 0] ** 2))
        right = float(np.sum(window[:, 1] ** 2))
        if pan < 0:
            assert left > right * 1.5, f"line at {start}s should favour the left ear"
        else:
            assert right > left * 1.5, f"line at {start}s should favour the right ear"


def test_ambience_survives_between_lines(mixed):
    """Silence between dubs must still carry the room, not be dead air."""
    audio, rate = sf.read(str(mixed / "dub_44k_stereo.wav"), dtype="float32")
    quiet = audio[int(10.5 * rate) : int(12.5 * rate)]
    rms = float(np.sqrt(np.mean(quiet**2)))
    assert rms > 1e-4, "ambience bed missing from the gap"


def test_mix_report_records_panning(mixed):
    with open(mixed / "mix_report.json", encoding="utf-8") as fh:
        report = json.load(fh)
    assert report["sample_rate"] == 44100
    assert report["segments"] == len(LINES)
    assert report["background_present"] is True
    assert report["panned_segments"] >= 3


def test_srt_is_chronological_and_complete(mixed):
    with open(mixed / "dub.srt", encoding="utf-8") as fh:
        blocks = [b for b in fh.read().strip().split("\n\n") if b.strip()]
    assert len(blocks) == len(LINES)
    sys.path.insert(0, ROOT)
    from asmrdub.srt_time import parse_time

    previous = -1.0
    for index, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        assert int(lines[0]) == index
        start_text, end_text = lines[1].split(" --> ")
        start, end = parse_time(start_text), parse_time(end_text)
        assert end > start
        assert start >= previous
        previous = start
        assert lines[2].strip()
