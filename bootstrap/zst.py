"""Extract a `.tar.zst` archive without assuming zstd is installed.

Kaggle's image has neither `zstd` nor `unzstd`, and ollama only publishes
`.tar.zst` now -- the older `.tgz` URL returns 404. Upstream's own install.sh
gives up at exactly this point and tells you to `apt-get install zstd`.

So this tries four things in order of decreasing predictability:

1. a `zstd`/`unzstd`/`pzstd` binary, piped straight into tar;
2. `zstandard` (or 3.14's stdlib `compression.zstd`) in *this* interpreter;
3. `zstandard` installed with uv into a Python we already built, then the same
   in-process path re-run there as a subprocess;
4. `apt-get install -y zstd`, then (1) again.

apt is last on purpose: it needs root and a working package index, and on a
notebook image it is the slowest and least predictable of the four.

Every path streams -- decompressed bytes go into `tarfile` through a pipe, never
onto disk. The ollama archive is 1.4GB compressed and about 2.6GB expanded, and
writing that intermediate would cost both time and quota for nothing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile

ZSTD_BINARIES = ("zstd", "unzstd", "pzstd")

# The ollama tree relies on relative symlinks between its bundled CUDA
# libraries, so extraction must keep links. "tar" still refuses absolute paths
# and `..` traversal; "data" would additionally drop the links we need.
TAR_FILTER = "tar"


def find_binary() -> str | None:
    """Path to a zstd CLI, or None."""
    for name in ZSTD_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def zstd_reader(fileobj):
    """A streaming zstd decompressor around `fileobj`.

    Prefers the third-party `zstandard`; falls back to `compression.zstd`, which
    is stdlib from Python 3.14. Raises ImportError when neither exists.
    """
    try:
        import zstandard

        return zstandard.ZstdDecompressor().stream_reader(fileobj)
    except ImportError:
        pass
    from compression.zstd import ZstdFile   # Python >= 3.14

    return ZstdFile(fileobj, "rb")


def have_module(python: str | None = None) -> bool:
    """Is a zstd decompressor importable -- here, or in another interpreter?"""
    probe = (
        "import sys\n"
        "try:\n"
        "    import zstandard\n"
        "except ImportError:\n"
        "    try:\n"
        "        import compression.zstd\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
    )
    if python is None:
        try:
            zstd_reader  # noqa: B018 - just proving the name resolves
            import zstandard  # noqa: F401

            return True
        except ImportError:
            try:
                import compression.zstd  # noqa: F401

                return True
            except ImportError:
                return False
    return subprocess.run([python, "-c", probe]).returncode == 0


def extract_with_module(archive: str, dest: str) -> None:
    """Stream the archive through an in-process zstd decompressor into tar."""
    os.makedirs(dest, exist_ok=True)
    with open(archive, "rb") as raw:
        reader = zstd_reader(raw)
        # "r|" is the streaming reader: it never seeks, which is all a
        # decompressor can offer.
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            try:
                tar.extractall(dest, filter=TAR_FILTER)
            except TypeError:
                # `filter` arrived in 3.11.4; older builds have no such kwarg.
                tar.extractall(dest)


def extract_with_binary(binary: str, archive: str, dest: str) -> None:
    """`zstd -d -c archive | tar -xf - -C dest`, without a shell.

    Built with two Popens rather than `shell=True` so the paths cannot be
    re-parsed by a shell -- they come from a scratch directory whose name we do
    not control.
    """
    os.makedirs(dest, exist_ok=True)
    decompress = subprocess.Popen(
        [binary, "-d", "-c", archive], stdout=subprocess.PIPE
    )
    try:
        untar = subprocess.Popen(
            ["tar", "-xf", "-", "-C", dest], stdin=decompress.stdout
        )
        # Let the writer see EPIPE if tar dies first.
        decompress.stdout.close()
        untar_rc = untar.wait()
    finally:
        decompress_rc = decompress.wait()
    if decompress_rc != 0:
        raise RuntimeError(f"{os.path.basename(binary)} failed (rc={decompress_rc})")
    if untar_rc != 0:
        raise RuntimeError(f"tar failed (rc={untar_rc})")


def extract_with_helper(python: str, archive: str, dest: str, uv: str | None) -> None:
    """Install `zstandard` into `python`'s environment, then extract there.

    `uv venv` environments have no pip, so uv is the only way to add a package
    to one. The subprocess re-imports *this* module rather than carrying a copy
    of the extraction logic in a string.
    """
    if not have_module(python):
        if not uv:
            raise RuntimeError("zstandard is missing and no uv is available to add it")
        result = subprocess.run(
            [uv, "pip", "install", "--python", python, "zstandard>=0.22"]
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not install zstandard into {python}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import sys\n"
        f"sys.path.insert(0, {root!r})\n"
        "from bootstrap.zst import extract_with_module\n"
        f"extract_with_module({archive!r}, {dest!r})\n"
    )
    result = subprocess.run([python, "-c", script])
    if result.returncode != 0:
        raise RuntimeError(f"helper extraction failed (rc={result.returncode})")


def apt_install_zstd() -> str | None:
    """Last resort: install the CLI. Returns its path, or None on failure."""
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    for command in (
        ["apt-get", "install", "-y", "-qq", "zstd"],
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "-qq", "zstd"],
    ):
        if subprocess.run(command, env=env).returncode == 0:
            found = find_binary()
            if found:
                return found
    return None


def extract(
    archive: str,
    dest: str,
    helper_python: str | None = None,
    uv: str | None = None,
    log=print,
) -> str:
    """Extract `archive` into `dest`. Returns the strategy that worked.

    Raises RuntimeError listing every attempt when all of them fail, so the
    notebook output says what to install rather than just "tar failed".
    """
    attempts: list[str] = []

    binary = find_binary()
    if binary:
        try:
            extract_with_binary(binary, archive, dest)
            return f"binary:{os.path.basename(binary)}"
        except Exception as exc:  # noqa: BLE001 - fall through to the next way
            attempts.append(f"{os.path.basename(binary)} pipe: {exc}")
            log(f"zstd binary failed ({exc}); trying a Python decompressor")
    else:
        attempts.append("no zstd/unzstd/pzstd on PATH")

    if have_module():
        try:
            extract_with_module(archive, dest)
            return "module:in-process"
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"in-process zstandard: {exc}")
            log(f"in-process decompression failed ({exc})")
    else:
        attempts.append("zstandard not importable in this interpreter")

    if helper_python:
        try:
            extract_with_helper(helper_python, archive, dest, uv)
            return f"helper:{helper_python}"
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"helper {helper_python}: {exc}")
            log(f"helper extraction failed ({exc}); trying apt-get")
    else:
        attempts.append("no helper interpreter offered")

    installed = apt_install_zstd()
    if installed:
        try:
            extract_with_binary(installed, archive, dest)
            return f"apt:{os.path.basename(installed)}"
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"apt-installed {installed}: {exc}")
    else:
        attempts.append("apt-get could not install zstd")

    raise RuntimeError(
        "could not extract %s -- every method failed:\n  - %s\n"
        "Fix by installing either: `apt-get install -y zstd` or "
        "`pip install zstandard`."
        % (os.path.basename(archive), "\n  - ".join(attempts))
    )


def describe_support(helper_python: str | None = None) -> str:
    """One line for the environment report, so a failure is predictable."""
    binary = find_binary()
    parts = [f"binary={os.path.basename(binary) if binary else 'none'}"]
    parts.append(f"module={'yes' if have_module() else 'no'}")
    if helper_python:
        parts.append(f"helper={'yes' if have_module(helper_python) else 'installable'}")
    return " · ".join(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="extract a .tar.zst archive")
    parser.add_argument("archive")
    parser.add_argument("dest")
    parser.add_argument("--helper-python")
    parser.add_argument("--uv")
    args = parser.parse_args()
    print(
        extract(
            args.archive, args.dest,
            helper_python=args.helper_python, uv=args.uv,
        )
    )
