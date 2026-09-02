"""Verify the worker's separation maths with real torchaudio, no GPU.

`do_separate` is the one worker handler with substantial logic of its own:
normalise by the whole track, run chunks, cross-fade, reassemble, save. A bug
there is inaudible in unit tests of `plan_chunks` alone but very audible in the
product (seams, level jumps, or a mono downmix).

So this runs the real handler on CPU with a stand-in "model" that returns known
stems. Skipped unless torch and torchaudio are importable.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

SR = 44100
SOURCES = ("drums", "bass", "other", "vocals")


def _load_worker():
    path = os.path.join(ROOT, "worker", "server.py")
    spec = importlib.util.spec_from_file_location("asmrdub_worker_sep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SplitModel(torch.nn.Module):
    """Stand-in for HDemucs: puts the high tone in vocals, the low in 'other'.

    Returns (batch, source, channel, time) exactly like hdemucs_high, so the
    handler's indexing and summing are exercised for real.
    """

    def forward(self, mixture):
        batch, channels, samples = mixture.shape
        out = torch.zeros(batch, len(SOURCES), channels, samples)
        # Crude split by sign of the first difference: enough to produce two
        # distinct, reconstructable stems.
        high = mixture - torch.nn.functional.avg_pool1d(
            mixture, kernel_size=9, stride=1, padding=4, count_include_pad=False
        )
        low = mixture - high
        out[:, SOURCES.index("vocals")] = high
        out[:, SOURCES.index("other")] = low
        return out


class _PassThrough(torch.nn.Module):
    """Puts the entire mixture into vocals; nothing into the other stems.

    Makes the reconstruction assertion exact: vocals must come back equal to
    the input, which is only true if normalise/denormalise and the crossfade
    are all correct.
    """

    def forward(self, mixture):
        batch, channels, samples = mixture.shape
        out = torch.zeros(batch, len(SOURCES), channels, samples)
        out[:, SOURCES.index("vocals")] = mixture
        return out


@pytest.fixture
def workspace(tmp_path):
    """A 25s stereo file: long enough to force several chunks at chunk_sec=5."""
    duration = 25.0
    samples = int(duration * SR)
    time = np.arange(samples) / SR
    left = 0.4 * np.sin(2 * np.pi * 220 * time) + 0.2 * np.sin(2 * np.pi * 3000 * time)
    right = 0.3 * np.sin(2 * np.pi * 220 * time) + 0.35 * np.sin(2 * np.pi * 3000 * time)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    source = tmp_path / "source.wav"
    sf.write(str(source), stereo, SR, subtype="PCM_16")
    return tmp_path, source, stereo


def _run(worker, tmp_path, source, model, chunk_sec=5.0, overlap=0.1):
    worker.STATE["log"] = None
    worker.STATE["device"] = "cpu"
    worker.STATE["sep"] = model
    worker.STATE["loaded"] = "sep"
    payload = {
        "asmrdub_path": ROOT,
        "input": str(source),
        "vocal_out": str(tmp_path / "vocal.wav"),
        "background_out": str(tmp_path / "background.wav"),
        "sample_rate": SR,
        "chunk_sec": chunk_sec,
        "overlap": overlap,
    }
    return worker.do_separate(payload), payload


def test_separate_writes_both_stems_at_full_rate(workspace):
    worker = _load_worker()
    tmp_path, source, _ = workspace
    reply, payload = _run(worker, tmp_path, source, _SplitModel())
    assert reply["ok"] is True
    assert reply["chunks"] > 1, "the test file should need several chunks"
    for key in ("vocal_out", "background_out"):
        info = sf.info(payload[key])
        assert info.samplerate == SR
        assert info.channels == 2
        assert info.duration == pytest.approx(reply["duration"], abs=0.01)


def test_passthrough_reconstructs_the_input_exactly(workspace):
    """The crossfade and the normalise/denormalise must be lossless together.

    If the seam gains did not sum to 1, or if std/mean were applied per chunk,
    this would show up as periodic level dips every chunk_sec seconds.

    Compares `vocal + background` rather than `vocal` alone because the mixture's
    DC offset is deliberately assigned to the background stem.
    """
    worker = _load_worker()
    tmp_path, source, original = workspace
    _, payload = _run(worker, tmp_path, source, _PassThrough())
    vocal, rate = sf.read(payload["vocal_out"], dtype="float32", always_2d=True)
    background, _ = sf.read(payload["background_out"], dtype="float32", always_2d=True)
    assert rate == SR
    assert vocal.shape == original.shape
    # PCM_16 quantisation is the only permitted difference.
    error = np.abs((vocal + background) - original).max()
    assert error < 2e-4, f"max reconstruction error {error}"


def test_no_periodic_seam_artefacts(workspace):
    """Level must be flat across chunk boundaries, not dipping every 5s."""
    worker = _load_worker()
    tmp_path, source, _ = workspace
    _, payload = _run(worker, tmp_path, source, _PassThrough(), chunk_sec=5.0)
    vocal, _ = sf.read(payload["vocal_out"], dtype="float32", always_2d=True)
    background, _ = sf.read(payload["background_out"], dtype="float32", always_2d=True)
    recovered = vocal + background
    window = SR // 10
    levels = [
        float(np.sqrt(np.mean(recovered[i : i + window] ** 2)))
        for i in range(0, len(recovered) - window, window)
    ]
    assert min(levels) > 0.5 * max(levels), (
        f"level varies too much across the track: {min(levels):.4f}..{max(levels):.4f}"
    )


def test_stems_sum_back_to_the_mixture(workspace):
    """vocal + background must equal the input; a dropped stem would not.

    This is also what pins down where the DC offset goes: the mixture mean is
    removed before separation, and if it were added back to every stem it would
    appear four times over here.
    """
    worker = _load_worker()
    tmp_path, source, original = workspace
    _, payload = _run(worker, tmp_path, source, _SplitModel())
    vocal, _ = sf.read(payload["vocal_out"], dtype="float32", always_2d=True)
    background, _ = sf.read(payload["background_out"], dtype="float32", always_2d=True)
    total = vocal + background
    error = np.abs(total - original).max()
    assert error < 5e-3, f"stems do not sum back to the mixture (max err {error})"


def test_dc_offset_is_not_duplicated_across_stems(tmp_path):
    """A file with a real DC offset: the offset must land on one stem only."""
    worker = _load_worker()
    samples = int(12.0 * SR)
    offset = 0.25
    tone = (0.2 * np.sin(2 * np.pi * 300 * np.arange(samples) / SR) + offset).astype(
        np.float32
    )
    stereo = np.stack([tone, tone], axis=1)
    source = tmp_path / "dc.wav"
    sf.write(str(source), stereo, SR, subtype="PCM_16")

    _, payload = _run(worker, tmp_path, source, _SplitModel())
    vocal, _ = sf.read(payload["vocal_out"], dtype="float32", always_2d=True)
    background, _ = sf.read(payload["background_out"], dtype="float32", always_2d=True)

    # Vocal (the high-frequency stem here) must carry no offset...
    assert abs(float(vocal.mean())) < 0.01, float(vocal.mean())
    # ...and the sum must still reproduce the offset exactly once.
    assert float((vocal + background).mean()) == pytest.approx(offset, abs=0.01)


def test_mono_input_is_promoted_to_stereo(tmp_path):
    """HDemucs needs 2 channels; a mono upload must not crash the stage."""
    worker = _load_worker()
    samples = int(12.0 * SR)
    mono = (0.3 * np.sin(2 * np.pi * 300 * np.arange(samples) / SR)).astype(np.float32)
    source = tmp_path / "mono.wav"
    sf.write(str(source), mono, SR, subtype="PCM_16")
    reply, payload = _run(worker, tmp_path, source, _PassThrough())
    assert reply["channels"] == 2
    assert sf.info(payload["vocal_out"]).channels == 2


def test_resamples_a_non_44k_input(tmp_path):
    """Users upload 48kHz files; the pipeline is 44.1kHz throughout."""
    worker = _load_worker()
    in_rate = 48000
    samples = int(12.0 * in_rate)
    tone = (0.3 * np.sin(2 * np.pi * 300 * np.arange(samples) / in_rate)).astype(
        np.float32
    )
    stereo = np.stack([tone, tone], axis=1)
    source = tmp_path / "48k.wav"
    sf.write(str(source), stereo, in_rate, subtype="PCM_16")
    reply, payload = _run(worker, tmp_path, source, _PassThrough())
    assert reply["sample_rate"] == SR
    assert sf.info(payload["vocal_out"]).samplerate == SR
    assert reply["duration"] == pytest.approx(12.0, abs=0.05)


def test_short_input_uses_a_single_chunk(tmp_path):
    worker = _load_worker()
    samples = int(3.0 * SR)
    tone = (0.3 * np.sin(2 * np.pi * 300 * np.arange(samples) / SR)).astype(np.float32)
    source = tmp_path / "short.wav"
    sf.write(str(source), np.stack([tone, tone], axis=1), SR, subtype="PCM_16")
    reply, _ = _run(worker, tmp_path, source, _PassThrough(), chunk_sec=5.0)
    assert reply["chunks"] == 1


def test_output_never_clips(workspace):
    """A loud input must be scaled, not wrapped, when a stem overshoots."""
    worker = _load_worker()
    tmp_path, _, _ = workspace
    samples = int(12.0 * SR)
    loud = (0.99 * np.sin(2 * np.pi * 300 * np.arange(samples) / SR)).astype(np.float32)
    source = tmp_path / "loud.wav"
    sf.write(str(source), np.stack([loud, loud], axis=1), SR, subtype="PCM_16")
    _, payload = _run(worker, tmp_path, source, _PassThrough())
    for key in ("vocal_out", "background_out"):
        data, _ = sf.read(payload[key], dtype="float32", always_2d=True)
        assert float(np.abs(data).max()) <= 1.0
