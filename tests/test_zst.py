"""Tests for `.tar.zst` extraction with no zstd anywhere.

Kaggle's image has neither the `zstd` binary nor the Python module, which is how
the real run died:

    tar (child): unzstd: Cannot exec: No such file or directory

This machine happens to be in the same state (no zstd binary, no `zstandard` in
the interpreter that runs the tests unless it was installed), so the fixtures
build a real zstd archive using a hand-written encoder: zstd frames may contain
*raw* (uncompressed) blocks, so a valid archive can be produced with nothing but
`struct`. It decompresses correctly with any real zstd implementation.

Each fallback path is then forced by monkeypatching the ones above it, so all
four are exercised rather than only whichever the local machine happens to have.
"""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
import struct
import subprocess
import sys
import tarfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bootstrap import zst  # noqa: E402


# --------------------------------------------------------------------------
# A minimal zstd encoder: raw blocks only.
# --------------------------------------------------------------------------

MAGIC = 0xFD2FB528
MAX_RAW_BLOCK = 1 << 17          # 128 KiB, the format's block ceiling


def zstd_raw_frame(payload: bytes) -> bytes:
    """A valid single-segment zstd frame carrying `payload` in raw blocks.

    Frame_Header_Descriptor 0x20 sets Single_Segment_Flag, so Window_Descriptor
    is omitted and Frame_Content_Size is present as 8 bytes (FCS_Field_Size=3
    would need descriptor bits; 0x20 gives size-field code 0 => 1 byte, so the
    content size is written per the spec's 1-byte form when it fits, else the
    8-byte form is selected with the top bits).
    """
    out = bytearray()
    out += struct.pack("<I", MAGIC)

    size = len(payload)
    if size < 256:
        # FCS_Field_Size = 1 (descriptor bits 6-7 = 00), single segment.
        out += bytes([0x20])
        out += struct.pack("<B", size)
    else:
        # FCS_Field_Size = 8 (descriptor bits 6-7 = 11), single segment.
        out += bytes([0xE0])
        out += struct.pack("<Q", size)

    if not payload:
        # A frame still needs one (empty, last) block.
        out += struct.pack("<I", (0 << 1) | 1 | (0 << 3))[:3]
        return bytes(out)

    offset = 0
    while offset < size:
        chunk = payload[offset : offset + MAX_RAW_BLOCK]
        offset += len(chunk)
        last = 1 if offset >= size else 0
        # Block_Header: 3 bytes LE = Last_Block | Block_Type<<1 | Size<<3
        header = last | (0 << 1) | (len(chunk) << 3)   # Block_Type 0 = Raw
        out += header.to_bytes(3, "little")
        out += chunk
    return bytes(out)


def make_tar_zst(path: str, entries: dict[str, bytes], symlinks=None) -> str:
    """Build a real `.tar.zst` at `path`."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o755 if name.endswith("ollama") else 0o644
            tar.addfile(info, io.BytesIO(content))
        for link_name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(link_name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(zstd_raw_frame(buffer.getvalue()))
    return path


OLLAMA_TREE = {
    "bin/ollama": b"#!/bin/sh\necho ollama\n",
    "lib/ollama/libggml-base.so": b"\x7fELF" + b"\0" * 64,
    "lib/ollama/cuda_v12/libggml-cuda.so": b"\x7fELF" + b"\0" * 64,
}


@pytest.fixture
def archive(tmp_path):
    return make_tar_zst(str(tmp_path / "ollama-linux-amd64.tar.zst"), OLLAMA_TREE)


# --------------------------------------------------------------------------
# The fixture itself must be a valid archive, or every test below is vacuous.
# --------------------------------------------------------------------------


def test_fixture_is_a_real_zstd_frame(archive):
    with open(archive, "rb") as fh:
        assert struct.unpack("<I", fh.read(4))[0] == MAGIC


def test_fixture_round_trips_through_a_real_decompressor(archive, tmp_path):
    """Proves the hand-rolled encoder is spec-correct, not just magic-numbered."""
    zstandard = pytest.importorskip(
        "zstandard", reason="need a real zstd implementation to validate the fixture"
    )
    with open(archive, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            names = tar.getnames()
    assert sorted(names) == sorted(OLLAMA_TREE)


def test_fixture_handles_a_payload_spanning_several_blocks(tmp_path):
    """The 128KiB block ceiling means a big archive must be split correctly."""
    zstandard = pytest.importorskip("zstandard")
    big = {"lib/big.so": os.urandom(400_000)}
    path = make_tar_zst(str(tmp_path / "big.tar.zst"), big)
    with open(path, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            member = tar.next()
            assert member.name == "lib/big.so"
            assert tar.extractfile(member).read() == big["lib/big.so"]


# --------------------------------------------------------------------------
# Path 2: in-process module
# --------------------------------------------------------------------------


def test_extract_with_module(archive, tmp_path):
    pytest.importorskip("zstandard")
    dest = tmp_path / "out-module"
    zst.extract_with_module(archive, str(dest))
    for name in OLLAMA_TREE:
        assert (dest / name).is_file(), name
    assert (dest / "bin" / "ollama").read_bytes() == OLLAMA_TREE["bin/ollama"]


def test_extract_preserves_symlinks(tmp_path):
    """ollama's bundled CUDA libs are relative symlinks; dropping them breaks it."""
    pytest.importorskip("zstandard")
    path = make_tar_zst(
        str(tmp_path / "links.tar.zst"),
        {"lib/real.so": b"\x7fELF"},
        symlinks={"lib/alias.so": "real.so"},
    )
    dest = tmp_path / "out-links"
    zst.extract_with_module(path, str(dest))
    alias = dest / "lib" / "alias.so"
    assert alias.is_symlink() or alias.is_file(), "symlink member was dropped"


def test_extract_rejects_path_traversal(tmp_path):
    """A malicious archive must not write outside dest (tarfile filter='tar')."""
    pytest.importorskip("zstandard")
    if sys.version_info < (3, 11, 4):
        pytest.skip("tarfile extraction filters need >= 3.11.4")
    path = make_tar_zst(str(tmp_path / "evil.tar.zst"), {"../escaped.txt": b"nope"})
    dest = tmp_path / "out-evil"
    with pytest.raises(Exception):
        zst.extract_with_module(path, str(dest))
    assert not (tmp_path / "escaped.txt").exists()


# --------------------------------------------------------------------------
# Path 1: binary pipe
# --------------------------------------------------------------------------


def _fake_zstd(tmp_path, behaviour="ok"):
    """A stand-in `zstd` CLI written in Python, so no real one is needed."""
    script = tmp_path / f"fake_zstd_{behaviour}.py"
    if behaviour == "ok":
        body = (
            "import sys, io, os\n"
            f"sys.path.insert(0, {ROOT!r})\n"
            "from bootstrap.zst import zstd_reader\n"
            "args = sys.argv[1:]\n"
            "path = args[-1]\n"
            "with open(path, 'rb') as fh:\n"
            "    reader = zstd_reader(fh)\n"
            "    out = sys.stdout.buffer\n"
            "    while True:\n"
            "        chunk = reader.read(1 << 16)\n"
            "        if not chunk:\n"
            "            break\n"
            "        out.write(chunk)\n"
            "    out.flush()\n"
        )
    else:
        body = "import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)\n"
    script.write_text(body, encoding="utf-8")

    launcher = tmp_path / ("zstd.cmd" if os.name == "nt" else "zstd")
    if os.name == "nt":
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        os.chmod(launcher, 0o755)
    return str(launcher)


@pytest.mark.skipif(not shutil.which("tar"), reason="needs a tar binary")
def test_extract_with_binary(archive, tmp_path):
    pytest.importorskip("zstandard")   # the fake zstd uses it internally
    binary = _fake_zstd(tmp_path, "ok")
    dest = tmp_path / "out-binary"
    zst.extract_with_binary(binary, archive, str(dest))
    for name in OLLAMA_TREE:
        assert (dest / name).is_file(), name


@pytest.mark.skipif(not shutil.which("tar"), reason="needs a tar binary")
def test_binary_failure_is_reported(archive, tmp_path):
    binary = _fake_zstd(tmp_path, "fail")
    with pytest.raises(RuntimeError):
        zst.extract_with_binary(binary, archive, str(tmp_path / "out-fail"))


# --------------------------------------------------------------------------
# The dispatcher: force each rung of the ladder
# --------------------------------------------------------------------------


def test_dispatcher_prefers_the_binary(archive, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(zst, "find_binary", lambda: "/fake/zstd")
    monkeypatch.setattr(
        zst, "extract_with_binary",
        lambda b, a, d: calls.append(("binary", b)),
    )
    monkeypatch.setattr(
        zst, "extract_with_module",
        lambda a, d: calls.append(("module", a)),
    )
    method = zst.extract(archive, str(tmp_path / "o"), log=lambda m: None)
    assert method.startswith("binary:")
    assert [name for name, _ in calls] == ["binary"]


def test_dispatcher_falls_back_to_the_module_when_no_binary(archive, tmp_path, monkeypatch):
    """The exact Kaggle situation: no zstd on PATH at all."""
    pytest.importorskip("zstandard")
    monkeypatch.setattr(zst, "find_binary", lambda: None)
    dest = tmp_path / "out-nobinary"
    method = zst.extract(archive, str(dest), log=lambda m: None)
    assert method == "module:in-process"
    assert (dest / "bin" / "ollama").is_file()


def test_dispatcher_falls_back_when_the_binary_pipe_fails(archive, tmp_path, monkeypatch):
    pytest.importorskip("zstandard")
    monkeypatch.setattr(zst, "find_binary", lambda: "/fake/zstd")

    def explode(binary, source, dest):
        raise RuntimeError("zstd died")

    monkeypatch.setattr(zst, "extract_with_binary", explode)
    dest = tmp_path / "out-pipefail"
    method = zst.extract(archive, str(dest), log=lambda m: None)
    assert method == "module:in-process"
    assert (dest / "bin" / "ollama").is_file()


def test_dispatcher_uses_the_helper_interpreter(archive, tmp_path, monkeypatch):
    """No binary and no module here, but a venv one uv-install away."""
    monkeypatch.setattr(zst, "find_binary", lambda: None)
    monkeypatch.setattr(zst, "have_module", lambda python=None: python is not None)
    used = {}

    def fake_helper(python, source, dest, uv):
        used["python"] = python
        used["uv"] = uv
        zst.extract_with_module(source, dest)

    monkeypatch.setattr(zst, "extract_with_helper", fake_helper)
    dest = tmp_path / "out-helper"
    method = zst.extract(
        archive, str(dest), helper_python="/venv/bin/python", uv="/usr/bin/uv",
        log=lambda m: None,
    )
    assert method == "helper:/venv/bin/python"
    assert used == {"python": "/venv/bin/python", "uv": "/usr/bin/uv"}


def test_dispatcher_tries_apt_last(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(zst, "find_binary", lambda: None)
    monkeypatch.setattr(zst, "have_module", lambda python=None: False)
    order = []

    def fake_apt():
        order.append("apt")
        return "/usr/bin/zstd"

    monkeypatch.setattr(zst, "apt_install_zstd", fake_apt)
    monkeypatch.setattr(
        zst, "extract_with_binary",
        lambda b, a, d: order.append(f"binary:{b}"),
    )
    method = zst.extract(archive, str(tmp_path / "o"), log=lambda m: None)
    assert method == "apt:zstd"
    assert order == ["apt", "binary:/usr/bin/zstd"]


def test_dispatcher_error_lists_every_attempt(archive, tmp_path, monkeypatch):
    """The whole point: a failure must say what to install, not 'tar failed'."""
    monkeypatch.setattr(zst, "find_binary", lambda: None)
    monkeypatch.setattr(zst, "have_module", lambda python=None: False)
    monkeypatch.setattr(zst, "apt_install_zstd", lambda: None)
    with pytest.raises(RuntimeError) as info:
        zst.extract(archive, str(tmp_path / "o"), log=lambda m: None)
    message = str(info.value)
    assert "no zstd/unzstd/pzstd on PATH" in message
    assert "not importable" in message
    assert "apt-get install -y zstd" in message
    assert "pip install zstandard" in message


def test_apt_is_not_attempted_when_earlier_paths_work(archive, tmp_path, monkeypatch):
    """apt needs root and a package index; it must stay a last resort."""
    pytest.importorskip("zstandard")
    monkeypatch.setattr(zst, "find_binary", lambda: None)
    monkeypatch.setattr(
        zst, "apt_install_zstd",
        lambda: pytest.fail("apt-get must not run when the module works"),
    )
    zst.extract(archive, str(tmp_path / "o"), log=lambda m: None)


# --------------------------------------------------------------------------
# Support reporting and helper subprocess wiring
# --------------------------------------------------------------------------


def test_describe_support_mentions_both_channels():
    text = zst.describe_support()
    assert "binary=" in text and "module=" in text


def test_describe_support_reports_a_helper():
    text = zst.describe_support(helper_python=sys.executable)
    assert "helper=" in text


def test_have_module_agrees_with_a_subprocess_probe():
    """have_module(python) shells out; make sure its verdict is real."""
    verdict = zst.have_module(sys.executable)
    probe = subprocess.run(
        [sys.executable, "-c", "import zstandard"], capture_output=True
    )
    assert verdict == (probe.returncode == 0)


def test_helper_script_can_import_this_module_by_path(archive, tmp_path):
    """The helper re-imports bootstrap.zst; that import must resolve standalone."""
    pytest.importorskip("zstandard")
    dest = tmp_path / "out-subproc"
    script = (
        "import sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "from bootstrap.zst import extract_with_module\n"
        f"extract_with_module({archive!r}, {str(dest)!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "bin" / "ollama").is_file()


def test_module_is_runnable_as_a_cli(archive, tmp_path):
    pytest.importorskip("zstandard")
    dest = tmp_path / "out-cli"
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "bootstrap", "zst.py"), archive, str(dest)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "bin" / "ollama").is_file()


# --------------------------------------------------------------------------
# install_ollama wiring
# --------------------------------------------------------------------------


def _prepare_env():
    path = os.path.join(ROOT, "bootstrap", "prepare_env.py")
    spec = importlib.util.spec_from_file_location("asmrdub_prepare_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_ollama_extracts_a_local_archive(tmp_path, monkeypatch):
    """No download: the archive is pre-placed, as a resumed run would find it."""
    pytest.importorskip("zstandard")
    env = _prepare_env()
    scratch = tmp_path / "scratch"
    make_tar_zst(str(scratch / "ollama-linux-amd64.tar.zst"), OLLAMA_TREE)
    monkeypatch.setattr(env.zst, "find_binary", lambda: None)

    binary = env.install_ollama(str(scratch), helper_python=sys.executable)
    assert os.path.isfile(binary)
    assert binary.endswith(os.path.join("bin", "ollama"))


def test_install_ollama_is_idempotent(tmp_path, monkeypatch):
    pytest.importorskip("zstandard")
    env = _prepare_env()
    scratch = tmp_path / "scratch"
    make_tar_zst(str(scratch / "ollama-linux-amd64.tar.zst"), OLLAMA_TREE)
    monkeypatch.setattr(env.zst, "find_binary", lambda: None)

    first = env.install_ollama(str(scratch), helper_python=sys.executable)
    monkeypatch.setattr(
        env.zst, "extract",
        lambda *a, **k: pytest.fail("must not re-extract an installed ollama"),
    )
    assert env.install_ollama(str(scratch), helper_python=sys.executable) == first


def test_truncated_download_is_deleted_not_kept(tmp_path, monkeypatch):
    """A short `.part` must not be promoted; otherwise the next run 'resumes' it."""
    env = _prepare_env()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        env.urllib.request, "urlopen",
        lambda url, timeout=None: FakeResponse(b"truncated"),
    )
    with pytest.raises(RuntimeError, match="truncated|expected"):
        env.install_ollama(str(scratch))
    archive = scratch / "ollama-linux-amd64.tar.zst"
    assert not archive.exists()
    assert not (scratch / "ollama-linux-amd64.tar.zst.part").exists()


def test_find_named_prefers_an_executable(tmp_path):
    env = _prepare_env()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = tmp_path / "a" / "ollama"
    plain.write_bytes(b"x")
    exe = tmp_path / "b" / "ollama"
    exe.write_bytes(b"x")
    os.chmod(exe, 0o755)

    found = env._find_named(str(tmp_path), "ollama")
    assert found is not None
    if os.name != "nt":   # every file looks executable to os.access on Windows
        assert found == str(exe)


def test_cloudflared_rejects_a_stub_download(tmp_path, monkeypatch):
    env = _prepare_env()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        env.urllib.request, "urlopen",
        lambda url, timeout=None: FakeResponse(b"404 not found"),
    )
    with pytest.raises(RuntimeError, match="only"):
        env.install_cloudflared(str(tmp_path))
    assert not (tmp_path / "bin" / "cloudflared").exists()
