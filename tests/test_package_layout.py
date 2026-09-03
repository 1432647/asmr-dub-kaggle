"""Regression tests for top-level package naming and importability.

These exist because of a real Kaggle failure:

    ImportError: cannot import name 'prepare_env' from 'setup'
                 (/usr/local/lib/python3.12/dist-packages/setup/__init__.py)

Two mistakes combined. The directory was named `setup`, which Kaggle's image
already provides as an installed distribution -- and it had no `__init__.py`, so
it was only a *namespace* package. A regular package beats a namespace package
no matter how early its parent sits on sys.path, so `sys.path.insert(0, ROOT)`
did not help and could not have.

The tests below reproduce that exact shadowing with a decoy package, so removing
an `__init__.py` or renaming a directory back to a colliding name fails here
instead of on Kaggle 30 seconds into a session.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories imported with `from <name> import ...` somewhere in the project.
IMPORTED_PACKAGES = ("asmrdub", "bootstrap", "runtime")

# Entry points run as scripts by the bootstrap chain, with a flag that exercises
# every module-level import and then exits without doing any work.
ENTRY_POINTS = (
    ("bootstrap/bootstrap_main.py", "--help"),
    ("bootstrap/apply_overlay.py", "--help"),
    ("bootstrap/prepare_env.py", "--help"),
    ("bootstrap/prepare_models.py", "--help"),
    ("runtime/run_all.py", "--help"),
    ("runtime/ollama_svc.py", "--help"),
    ("worker/server.py", "--help"),
)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("package", IMPORTED_PACKAGES)
def test_imported_package_has_an_init(package):
    """No namespace packages. A namespace package loses to any installed one."""
    init = os.path.join(ROOT, package, "__init__.py")
    assert os.path.isfile(init), (
        f"{package}/__init__.py is missing -- it would become a namespace "
        f"package and any installed distribution named '{package}' would "
        f"shadow it (this is exactly how the old setup/ directory broke)"
    )


@pytest.mark.parametrize("package", IMPORTED_PACKAGES)
def test_package_name_is_not_already_installed(package):
    """The name must not collide with anything in this interpreter's stdlib/site.

    `setup` collided on Kaggle. This catches the same class of mistake against
    whatever is installed here, which is a weaker but still useful signal.
    """
    assert package not in sys.stdlib_module_names, f"{package} shadows a stdlib module"

    for key in ("purelib", "platlib", "stdlib"):
        location = sysconfig.get_paths().get(key)
        if not location or not os.path.isdir(location):
            continue
        for candidate in (
            os.path.join(location, package, "__init__.py"),
            os.path.join(location, package + ".py"),
        ):
            assert not os.path.exists(candidate), (
                f"'{package}' is already an installed package at {candidate}; "
                f"pick a name nobody else claims"
            )


def test_the_old_colliding_name_is_gone():
    """`setup/` must never come back: Kaggle's image owns that name."""
    assert not os.path.isdir(os.path.join(ROOT, "setup")), (
        "a top-level setup/ directory is shadowed by Kaggle's installed "
        "'setup' distribution; this package is called bootstrap/ for that reason"
    )


# --------------------------------------------------------------------------
# Behaviour: reproduce the shadowing and prove ours still wins
# --------------------------------------------------------------------------


def _decoy_tree(tmp_path):
    """A directory holding real (non-namespace) packages named like ours.

    Stands in for Kaggle's `dist-packages/setup/`. Importing anything from these
    raises ImportError, so if a decoy wins the import the test fails loudly with
    the same message the Kaggle run produced.
    """
    decoy = tmp_path / "decoy-site-packages"
    for package in IMPORTED_PACKAGES:
        directory = decoy / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text(
            "# decoy: stands in for an unrelated installed distribution\n"
            "DECOY = True\n",
            encoding="utf-8",
        )
    return decoy


@pytest.mark.parametrize("relative,flag", ENTRY_POINTS)
def test_entry_point_imports_under_shadowing(relative, flag, tmp_path):
    """Each script must import our packages even with decoys on the path."""
    decoy = _decoy_tree(tmp_path)
    env = {
        **os.environ,
        "PYTHONPATH": str(decoy),
        "PYTHONIOENCODING": "utf-8",
        # kaggle_dirs() probes for writability; keep it inside tmp_path.
        "ASMRDUB_HOME": str(tmp_path / "home"),
    }
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, *relative.split("/")), flag],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=ROOT, env=env, timeout=120,
    )
    combined = result.stdout + result.stderr
    assert "DECOY" not in combined
    assert "ImportError" not in combined, combined[-2000:]
    assert "ModuleNotFoundError" not in combined, combined[-2000:]
    # --help exits 0 after printing usage.
    assert result.returncode == 0, combined[-2000:]
    assert "usage:" in combined.lower(), combined[-2000:]


def test_decoy_would_actually_win_without_an_init(tmp_path):
    """Proves the decoy fixture is strong enough to catch the regression.

    Without this, the tests above could pass for the wrong reason (a decoy that
    never had a chance). Here the decoy is placed *first* on sys.path, and must
    win -- confirming that name shadowing is real and the fixture reproduces it.
    """
    decoy = _decoy_tree(tmp_path)
    probe = "import bootstrap, sys; print(getattr(bootstrap, 'DECOY', False))"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(tmp_path), env={**os.environ, "PYTHONPATH": str(decoy)}, timeout=60,
    )
    assert result.stdout.strip() == "True", (
        f"the decoy fixture is not shadowing anything: {result.stdout}{result.stderr}"
    )


def test_worker_can_be_imported_by_path_while_shadowed(tmp_path):
    """The worker is loaded by absolute path, not as a package.

    run_all.py launches it with the GPU venv's interpreter, so it must not need
    the repo importable as a package beyond the sys.path.insert it does itself.
    """
    decoy = _decoy_tree(tmp_path)
    probe = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('w', r'{os.path.join(ROOT, 'worker', 'server.py')}');"
        "m = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        "print('ROUTES', sorted(m.ROUTES))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(tmp_path), env={**os.environ, "PYTHONPATH": str(decoy)}, timeout=60,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    assert "/tts" in result.stdout and "/asr" in result.stdout
