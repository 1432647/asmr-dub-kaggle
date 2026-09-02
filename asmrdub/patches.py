"""Surgical patches applied to the pinned upstream VideoLingo checkout.

Design rule: prefer *adding* an overlay module over *editing* upstream. A new
file cannot break when upstream changes; an edit can. So only four edits
survive here, each because the behaviour lives inside a function we otherwise
want to reuse verbatim.

Every patch is (relative_path, old, new, why). ``old`` must appear exactly once
in the pinned revision -- ``apply_patches`` fails loudly rather than guessing,
because a silently skipped patch produces a confusing failure much later.
"""

from __future__ import annotations

import os
from typing import NamedTuple


class Patch(NamedTuple):
    path: str
    old: str
    new: str
    why: str


PATCHES: tuple[Patch, ...] = (
    # ------------------------------------------------------------------
    # 1-3. estimate_duration.py: make English g2p lazy.
    #
    # AdvancedSyllableEstimator.__init__ constructs G2p() eagerly. g2p_en
    # imports nltk and downloads averaged_perceptron_tagger on first use, so
    # a transient network failure kills the dubbing stage -- for a code path
    # that never runs here, since this pipeline is Japanese -> Chinese.
    # ------------------------------------------------------------------
    Patch(
        path="core/tts_backend/estimate_duration.py",
        old="from g2p_en import G2p\n",
        new=(
            "\n"
            "# asmr-dub: import g2p_en lazily. It pulls nltk and downloads tagger\n"
            "# data on construction, which only English syllable counting needs.\n"
            "def _load_g2p():\n"
            "    from g2p_en import G2p\n"
            "    return G2p()\n"
        ),
        why="avoid nltk download on the dubbing critical path",
    ),
    Patch(
        path="core/tts_backend/estimate_duration.py",
        old="        self.g2p_en = G2p()\n",
        new="        self._g2p_en = None\n",
        why="defer G2p construction",
    ),
    Patch(
        path="core/tts_backend/estimate_duration.py",
        old="    def estimate_duration(self, text: str, lang: Optional[str] = None) -> float:\n",
        new=(
            "    @property\n"
            "    def g2p_en(self):\n"
            "        if self._g2p_en is None:\n"
            "            self._g2p_en = _load_g2p()\n"
            "        return self._g2p_en\n"
            "\n"
            "    def estimate_duration(self, text: str, lang: Optional[str] = None) -> float:\n"
        ),
        why="expose g2p_en as a lazy property so callers are unchanged",
    ),
    # ------------------------------------------------------------------
    # 4. tts_main: forward the line number and task frame to custom_tts.
    #
    # Upstream calls custom_tts(text, save_as) only. Our backend needs the
    # line number to find that line's reference clip (refers/{number}.wav) and
    # the frame to read the original Japanese for prompt text. The alternative
    # -- parsing the number back out of the temp filename -- silently
    # mis-clones if upstream ever renames its template.
    # ------------------------------------------------------------------
    Patch(
        path="core/tts_backend/tts_main.py",
        old="                custom_tts(text, save_as)\n",
        new="                custom_tts(text, save_as, number, task_df)\n",
        why="custom_tts needs per-line reference audio",
    ),
    # ------------------------------------------------------------------
    # 5. Serialise TTS calls.
    #
    # The GPU worker holds a single lock, so parallel requests only queue and
    # risk client-side timeouts. gpt_sovits is already special-cased for the
    # same reason.
    # ------------------------------------------------------------------
    Patch(
        path="core/_10_gen_audio.py",
        old='        max_workers = load_key("max_workers") if load_key("tts_method") != "gpt_sovits" else 1\n',
        new=(
            "        # asmr-dub: custom_tts talks to a single-lock GPU worker;\n"
            "        # concurrency here only adds queueing and timeout risk.\n"
            '        _serial_tts = ("gpt_sovits", "custom_tts")\n'
            '        max_workers = 1 if load_key("tts_method") in _serial_tts else load_key("max_workers")\n'
        ),
        why="one TTS request at a time",
    ),
    # ------------------------------------------------------------------
    # 6. real_dur must be a float column.
    #
    # Upstream initialises it with `= 0`, giving an int64 column, then assigns
    # measured durations (2.78) into it. pandas <3 silently upcast; pandas 3
    # raises "Invalid value '2.78' for dtype 'int64'" and kills the dubbing
    # stage on its very first line. Upstream pins `pandas>=2.2.3` with no
    # ceiling, so any fresh install lands on 3.x and always hits this.
    # ------------------------------------------------------------------
    Patch(
        path="core/_10_gen_audio.py",
        old="    tasks_df['real_dur'] = 0\n",
        new="    tasks_df['real_dur'] = 0.0  # asmr-dub: float; pandas>=3 will not upcast\n",
        why="pandas 3 rejects float-into-int64 assignment",
    ),
)


def apply_patches(repo_root: str, patches=PATCHES) -> list[str]:
    """Apply every patch under ``repo_root``.

    Returns the list of "path: why" strings applied. Raises RuntimeError when
    an anchor is missing or ambiguous -- an unpinned upstream is a setup bug,
    not something to work around at runtime.
    """
    applied = []
    for patch in patches:
        target = os.path.join(repo_root, patch.path)
        if not os.path.isfile(target):
            raise RuntimeError(f"patch target missing: {patch.path}")
        with open(target, "r", encoding="utf-8") as fh:
            text = fh.read()
        if patch.new in text and patch.old not in text:
            applied.append(f"{patch.path}: already applied ({patch.why})")
            continue
        count = text.count(patch.old)
        if count != 1:
            raise RuntimeError(
                f"patch anchor found {count} times (expected 1) in {patch.path}\n"
                f"  anchor: {patch.old!r}\n"
                f"  upstream pin probably moved"
            )
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(text.replace(patch.old, patch.new, 1))
        applied.append(f"{patch.path}: {patch.why}")
    return applied


# Packages in upstream requirements.txt that must NOT be installed into the
# CPU-only app environment. Each of these drags in torch (or a whole torch
# stack), which belongs exclusively to the GPU worker venv.
APP_REQUIREMENT_BLOCKLIST = frozenset(
    {
        "whisperx",          # torch~=2.8 + torchvision + torchcodec
        "pyannote-audio",    # torch, torchaudio, torchmetrics
        "pytorch-lightning",
        "lightning",
    }
)

# Extra packages the app environment needs that upstream does not list.
APP_EXTRA_REQUIREMENTS = (
    "faster-whisper==1.2.1",   # CTranslate2 ASR client-side helpers
    "soundfile>=0.12",         # reference-clip slicing without torchaudio
    "sudachipy>=0.6.8",        # required by spaCy's Japanese tokenizer
    "sudachidict-core>=20240109",
)

# Requirements whose upstream specifier must be *replaced*, not supplemented.
# Appending a second line for the same distribution resolves correctly with uv
# but reads like a conflict; rewriting the one line says what we mean.
APP_REQUIREMENT_OVERRIDES = {
    # Upstream declares `pandas>=2.2.3` with no ceiling but was written against
    # pandas 2 semantics: it assigns floats into columns it initialised with int
    # literals, which pandas 3 rejects outright. We patch the instance we found
    # (_10_gen_audio's real_dur), and cap the version so an unpatched instance
    # in a stage we cannot exercise offline does not surface on Kaggle instead.
    "pandas": "pandas>=2.2.3,<3",
}


def filter_requirements(lines) -> list[str]:
    """Drop blocklisted requirements and apply overrides, preserving order.

    Comparison is on the normalised distribution name, so "pyannote-audio",
    "pyannote_audio" and "pyannote-audio>=4.0.0" all match.
    """
    kept = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = _requirement_name(line)
        if name in APP_REQUIREMENT_BLOCKLIST:
            continue
        kept.append(APP_REQUIREMENT_OVERRIDES.get(name, line))
    return kept


def _requirement_name(line: str) -> str:
    import re

    head = line.split(";", 1)[0].strip()
    name = re.split(r"\s*(?:==|>=|<=|~=|!=|>|<|\[|@)", head, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")
