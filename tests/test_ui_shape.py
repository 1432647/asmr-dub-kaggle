"""Import-and-shape tests for the Streamlit UI, without a browser.

Streamlit apps are notoriously untested: `streamlit run` executes the module
top-to-bottom, so a NameError in a rarely-visited branch only surfaces when a
user clicks it -- on Kaggle, 40 minutes into a session.

These tests do two things:

1. Import `asmr_ui` for real, with a stub `streamlit` module recording every
   widget call. That exercises every import, the module-level tee, and the
   password gate's code path.
2. Assert structural properties by AST: that every `core.*` symbol the UI calls
   actually exists upstream, and that the password gate cannot be bypassed.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VL_ROOT = os.environ.get("ASMRDUB_VL_ROOT", "")
UI_PATH = os.path.join(ROOT, "overlay", "asmr_ui.py")

pytestmark = pytest.mark.skipif(
    not VL_ROOT or not os.path.isdir(VL_ROOT),
    reason="set ASMRDUB_VL_ROOT to a patched VideoLingo checkout",
)


# --------------------------------------------------------------------------
# A stub streamlit good enough to import the app
# --------------------------------------------------------------------------


class _Recorder:
    """Records attribute access and calls; returns itself for chaining."""

    def __init__(self, log, name="st"):
        self._log = log
        self._name = name

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return _Recorder(self._log, f"{self._name}.{item}")

    def __call__(self, *args, **kwargs):
        self._log.append((self._name, args, kwargs))
        return _Widget(self._log, self._name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Widget(_Recorder):
    """A widget result: falsy so no button branch is taken during import."""

    def __bool__(self):
        return False

    def __iter__(self):
        # st.columns(n) is unpacked; yield enough recorders.
        return iter([_Widget(self._log, self._name) for _ in range(4)])

    def __getitem__(self, index):
        return _Widget(self._log, f"{self._name}[{index}]")


def _make_streamlit(log, session_state=None):
    module = types.ModuleType("streamlit")
    recorder = _Recorder(log)

    for name in dir(recorder):
        pass  # nothing to copy; __getattr__ handles everything

    module.__getattr__ = lambda item: getattr(recorder, item)  # type: ignore[attr-defined]
    module.session_state = {} if session_state is None else session_state

    def fragment(*fargs, **fkwargs):
        """Mimic @st.fragment(run_every=...) -- must return a decorator."""
        def decorate(func):
            return func
        if fargs and callable(fargs[0]):
            return fargs[0]
        return decorate

    module.fragment = fragment
    module.column_config = _Recorder(log, "st.column_config")
    module.set_page_config = lambda **kwargs: log.append(("set_page_config", (), kwargs))
    module.rerun = lambda **kwargs: log.append(("rerun", (), kwargs))
    module.stop = lambda: None
    return module


@pytest.fixture
def ui(monkeypatch, tmp_path):
    """Import overlay/asmr_ui.py with a stubbed streamlit, from a real checkout."""
    log: list = []
    session: dict = {}
    monkeypatch.setitem(sys.modules, "streamlit", _make_streamlit(log, session))

    monkeypatch.setenv("ASMRDUB_PASSWORD", "test-password")
    monkeypatch.setenv("ASMRDUB_LOG", str(tmp_path / "pipeline.log"))
    monkeypatch.setenv("ASMRDUB_PKG_PATH", ROOT)
    monkeypatch.setenv("ASMRDUB_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.syspath_prepend(VL_ROOT)
    monkeypatch.chdir(VL_ROOT)

    for name in list(sys.modules):
        if name == "asmr_ui" or name.startswith("core"):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location("asmr_ui", UI_PATH)
    module = importlib.util.module_from_spec(spec)
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        spec.loader.exec_module(module)
        # Record whether the module installed its tee, before we restore ours:
        # pytest's own capture replaces sys.stdout, so the tee cannot be left in
        # place for the rest of the test session.
        module.TEE_INSTALLED = isinstance(sys.stdout, module._Tee)
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
    return module, log, session


# --------------------------------------------------------------------------
# Import-level
# --------------------------------------------------------------------------


def test_ui_imports_and_renders_the_login_gate(ui):
    """Every import resolves, and an unauthenticated visit stops at the gate."""
    module, log, session = ui
    names = [entry[0] for entry in log]
    assert "st.set_page_config" in names or "set_page_config" in names
    assert any("text_input" in name for name in names), "no password field rendered"
    # The gate returned False, so no pipeline section was rendered.
    assert not any("data_editor" in name for name in names)


def test_missing_password_env_refuses_to_run(monkeypatch, tmp_path):
    """No password set must be a hard refusal, not an open door."""
    log: list = []
    monkeypatch.setitem(sys.modules, "streamlit", _make_streamlit(log, {}))
    monkeypatch.delenv("ASMRDUB_PASSWORD", raising=False)
    monkeypatch.setenv("ASMRDUB_LOG", str(tmp_path / "log.txt"))
    monkeypatch.setenv("ASMRDUB_PKG_PATH", ROOT)
    monkeypatch.syspath_prepend(VL_ROOT)
    monkeypatch.chdir(VL_ROOT)
    for name in list(sys.modules):
        if name == "asmr_ui" or name.startswith("core"):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location("asmr_ui_nopass", UI_PATH)
    module = importlib.util.module_from_spec(spec)
    saved = sys.stdout, sys.stderr
    try:
        spec.loader.exec_module(module)
    finally:
        sys.stdout, sys.stderr = saved

    errors = [entry for entry in log if entry[0] == "st.error"]
    assert errors, "no error shown when ASMRDUB_PASSWORD is unset"
    assert any("ASMRDUB_PASSWORD" in str(entry[1]) for entry in errors)


def test_stdout_is_teed_to_the_log_file(ui):
    """The log panel reads this file; without the tee it stays empty.

    Writes through the tee object the module actually installed, rather than
    through `print`: pytest swaps `sys.stdout` for its own capture object, so a
    bare print in a test never reaches the app's tee.
    """
    module, _, _ = ui
    assert module.TEE_INSTALLED, "asmr_ui did not install its stdout tee"

    tee = module._Tee(open(os.devnull, "w"), module.LOG_PATH)
    tee.write("hello from the pipeline\n")
    tee.flush()

    assert os.path.isfile(module.LOG_PATH)
    with open(module.LOG_PATH, encoding="utf-8") as fh:
        assert "hello from the pipeline" in fh.read()


def test_tee_writes_reach_both_destinations(ui, tmp_path):
    """Notebook stdout must keep working; the browser is an addition, not a swap."""
    module, _, _ = ui

    class Sink:
        def __init__(self):
            self.text = ""

        def write(self, data):
            self.text += data

        def flush(self):
            pass

    sink = Sink()
    path = str(tmp_path / "tee.log")
    tee = module._Tee(sink, path)
    tee.write("both places\n")
    tee.flush()

    assert "both places" in sink.text
    with open(path, encoding="utf-8") as fh:
        assert "both places" in fh.read()
    assert tee.isatty() is False, "rich would emit ANSI escapes into the log file"


def test_dataset_scan_finds_audio_and_ignores_other_files(ui, tmp_path):
    module, _, _ = ui
    root = tmp_path / "input" / "my-dataset"
    root.mkdir(parents=True)
    (root / "track.wav").write_bytes(b"\0")
    (root / "notes.txt").write_text("nope")
    (root / "nested").mkdir()
    (root / "nested" / "second.flac").write_bytes(b"\0")

    found = module.dataset_audio_files()
    assert any(path.endswith("track.wav") for path in found)
    assert any(path.endswith("second.flac") for path in found)
    assert not any(path.endswith("notes.txt") for path in found)


def test_dataset_scan_survives_a_missing_mount(ui):
    module, _, _ = ui
    os.environ["ASMRDUB_INPUT_ROOT"] = "/definitely/not/mounted"
    assert module.dataset_audio_files() == []


def test_step_lists_are_callable(ui):
    """A typo'd stage name here would raise only when the button is pressed."""
    module, _, _ = ui
    for label, func in module.text_steps() + module.audio_steps():
        assert isinstance(label, str) and label
        assert callable(func), label


# --------------------------------------------------------------------------
# Structural: does the UI call things that actually exist?
# --------------------------------------------------------------------------


def _ui_tree():
    with open(UI_PATH, encoding="utf-8") as fh:
        return ast.parse(fh.read())


def test_every_core_import_exists_upstream():
    """`from core import _2_asr, ...` -- each imported name must really exist.

    Handles both shapes: `from core import <module>` and
    `from core.utils import <function>`.
    """
    tree = _ui_tree()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_name = node.module or ""
        if not module_name.startswith("core"):
            continue

        parts = module_name.split(".")
        package_dir = os.path.join(VL_ROOT, *parts)
        module_file = package_dir + ".py"

        for alias in node.names:
            # (a) the name is a submodule of a package
            if os.path.isdir(package_dir):
                if os.path.isfile(os.path.join(package_dir, alias.name + ".py")):
                    continue
                if os.path.isdir(os.path.join(package_dir, alias.name)):
                    continue

            # (b) the name is defined in the module/package __init__
            source_path = (
                module_file
                if os.path.isfile(module_file)
                else os.path.join(package_dir, "__init__.py")
            )
            assert os.path.isfile(source_path), f"{module_name} does not exist"
            with open(source_path, encoding="utf-8") as fh:
                sub_tree = ast.parse(fh.read())
            defined = set()
            for sub in ast.walk(sub_tree):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            defined.add(target.id)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for sub_alias in sub.names:
                        defined.add(sub_alias.asname or sub_alias.name.split(".")[0])
            assert alias.name in defined, (
                f"{module_name}.{alias.name} does not exist upstream"
            )


def test_pipeline_functions_exist_in_upstream_modules():
    """Every `_2_asr.transcribe`-style call is checked against the real module."""
    expected = {
        "_2_asr": ["transcribe"],
        "_3_1_split_nlp": ["split_by_spacy"],
        "_3_2_split_meaning": ["split_sentences_by_meaning"],
        "_4_1_summarize": ["get_summary"],
        "_4_2_translate": ["translate_all"],
        "_5_split_sub": ["split_for_sub_main"],
        "_6_gen_sub": ["align_timestamp_main"],
        "_8_1_audio_task": ["gen_audio_task_main"],
        "_8_2_dub_chunks": ["gen_dub_chunks"],
        "_9_refer_audio": ["extract_refer_audio_main"],
        "_10_gen_audio": ["gen_audio"],
        "_11_merge_audio": ["merge_full_audio"],
    }
    for module_name, functions in expected.items():
        path = os.path.join(VL_ROOT, "core", module_name + ".py")
        assert os.path.isfile(path), path
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for function in functions:
            assert function in defined, f"{module_name}.{function} missing"


def test_config_keys_read_by_the_ui_exist():
    """load_key raises KeyError on an unknown key, killing the sidebar."""
    import ruamel.yaml

    from asmrdub.vl_config import flatten

    with open(os.path.join(VL_ROOT, "config.yaml"), encoding="utf-8") as fh:
        config = ruamel.yaml.YAML().load(fh)
    existing = {key for key, _ in flatten(config)}

    tree = _ui_tree()
    read = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("load_key", "update_key")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            read.add(node.args[0].value)
    assert read, "no config keys found in the UI (test is not doing its job)"
    assert read <= existing, f"unknown config keys: {sorted(read - existing)}"


def test_no_pipeline_section_runs_before_authentication():
    """main() must return early on a failed gate; ordering is the whole defence."""
    tree = _ui_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            first = node.body[0]
            assert isinstance(first, ast.If), "main() must start with the gate check"
            source = ast.dump(first)
            assert "require_password" in source
            assert "Return" in source
            break
    else:
        pytest.fail("main() not found")


def test_worker_and_llm_are_never_addressed_over_a_public_url():
    """A tunnelled worker URL would expose an unauthenticated GPU endpoint."""
    with open(UI_PATH, encoding="utf-8") as fh:
        source = fh.read()
    assert "trycloudflare" not in source
    assert "0.0.0.0:7861" not in source
