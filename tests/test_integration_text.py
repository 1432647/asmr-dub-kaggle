"""Test the whole text half of the pipeline against a fake OpenAI-compatible LLM.

This covers ASR-result -> spaCy split -> LLM meaning split -> terminology ->
translation -> subtitle split -> timestamp alignment -> dubbing task table.
Every one of those stages talks to `ask_gpt`, which is exactly the route ollama
serves, so a mismatch in prompt handling or JSON shape shows up here rather than
25 minutes into a Kaggle run.

The fake LLM parses each prompt to work out which stage is asking and answers in
that stage's required schema. It "translates" by prefixing, which is enough:
what is being tested is the plumbing and the alignment maths, not the wording.
"""

from __future__ import annotations

import ast
import json
import os
import re
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

pd = pytest.importorskip("pandas")
pytest.importorskip("spacy")
pytest.importorskip("openai")

# A short ASMR-ish transcript with the awkward bits that matter: a one-word
# interjection, a long run-on sentence, and repeated particles.
WORDS = [
    ("こんばんは", 1.00, 1.80),
    ("、", 1.80, 1.85),
    ("よく", 1.90, 2.30),
    ("眠れ", 2.30, 2.70),
    ("まし", 2.70, 2.95),
    ("た", 2.95, 3.10),
    ("か", 3.10, 3.40),
    ("。", 3.40, 3.50),
    ("ふふっ", 4.10, 4.80),
    ("。", 4.80, 4.90),
    ("今日", 5.60, 6.00),
    ("は", 6.00, 6.15),
    ("とても", 6.15, 6.70),
    ("静か", 6.70, 7.20),
    ("です", 7.20, 7.55),
    ("ね", 7.55, 7.90),
    ("。", 7.90, 8.00),
    ("また", 12.00, 12.40),
    ("明日", 12.40, 12.90),
    ("ね", 12.90, 13.20),
    ("、", 13.20, 13.30),
    ("おやすみ", 13.40, 14.10),
    ("なさい", 14.10, 14.70),
    ("。", 14.70, 14.80),
]


class _FakeLLM(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible endpoint that satisfies every prompt schema."""

    prompts: list[str] = []

    def do_POST(self):  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        prompt = request["messages"][-1]["content"]
        type(self).prompts.append(prompt)

        content = self._answer(prompt)
        body = json.dumps(
            {
                "id": "fake",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- prompt dispatch ------------------------------------------------
    def _answer(self, prompt: str) -> str:
        if "professional Netflix subtitle splitter" in prompt:
            return self._split(prompt)
        if "terminology consultant" in prompt:
            return json.dumps({"theme": "一段安静的ASMR对话。主要是问候和道别。",
                               "terms": []}, ensure_ascii=False)
        if "faithfully translating" in prompt or "Faithful to the original" in prompt:
            return self._translate(prompt, key="direct")
        if "optimizing the" in prompt or "reflect" in prompt.lower():
            return self._translate(prompt, key="free")
        if "align" in prompt.lower():
            return self._align(prompt)
        return json.dumps({"result": "ok"}, ensure_ascii=False)

    @staticmethod
    def _tagged(prompt: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", prompt, re.S)
        return match.group(1).strip() if match else ""

    def _split(self, prompt: str) -> str:
        sentence = self._tagged(prompt, "split_this_sentence")
        parts_match = re.search(r"into \*\*(\d+)\*\* parts", prompt)
        parts = int(parts_match.group(1)) if parts_match else 2
        # Split on character count so the [br] positions are always findable.
        step = max(1, len(sentence) // parts)
        pieces = [sentence[i * step:(i + 1) * step] for i in range(parts - 1)]
        pieces.append(sentence[(parts - 1) * step:])
        joined = "[br]".join(piece for piece in pieces if piece)
        return "```json\n" + json.dumps(
            {"analysis": "fake", "split1": joined, "split2": joined,
             "assess": "fake", "choice": "1"},
            ensure_ascii=False,
        ) + "\n```"

    def _translate(self, prompt: str, key: str) -> str:
        subtitles = self._tagged(prompt, "subtitles")
        if subtitles:
            lines = [line for line in subtitles.split("\n") if line.strip()]
        else:
            # The expressiveness prompt embeds the previous JSON instead.
            lines = re.findall(r'"origin":\s*"([^"]*)"', prompt)
        result = {}
        for index, line in enumerate(lines, start=1):
            result[str(index)] = {
                "origin": line,
                "direct": f"【中】{line}",
                "reflect": "fake",
                "free": f"【中】{line}",
            }
            if key == "direct":
                result[str(index)].pop("free")
                result[str(index)].pop("reflect")
        return "```json\n" + json.dumps(result, ensure_ascii=False) + "\n```"

    def _align(self, prompt: str) -> str:
        source_parts = self._tagged(prompt, "source_part") or self._tagged(
            prompt, "src_part"
        )
        count = max(2, len([p for p in source_parts.split("\n") if p.strip()]))
        return "```json\n" + json.dumps(
            {"align": [{f"target_part_{i + 1}": f"【中】片段{i + 1}"} for i in range(count)]},
            ensure_ascii=False,
        ) + "\n```"

    def log_message(self, *args):
        """Quiet."""


@pytest.fixture(scope="module")
def fake_llm():
    _FakeLLM.prompts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", _FakeLLM
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def repo(tmp_path_factory, fake_llm):
    """A checkout with a cleaned_chunks.xlsx, as _2_asr would leave it."""
    url, _ = fake_llm
    base = tmp_path_factory.mktemp("text")
    target = base / "VideoLingo"
    shutil.copytree(VL_ROOT, target, ignore=shutil.ignore_patterns("__pycache__", ".git"))

    log_dir = target / "output" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = target / "output" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # save_results() quotes each word; downstream strips the quotes back off.
    pd.DataFrame(
        [
            {"text": f'"{word}"', "start": start, "end": end, "speaker_id": None}
            for word, start, end in WORDS
        ]
    ).to_excel(str(log_dir / "cleaned_chunks.xlsx"), index=False)

    # A raw mix long enough for the trailing-gap computation.
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    sf.write(str(audio_dir / "raw.mp3"), np.zeros(16000 * 20, dtype="float32"), 16000)

    _point_config_at(target, url)
    return target


def _point_config_at(repo, url: str) -> None:
    """Repoint config.yaml at the fake LLM, exactly as setup does for ollama."""
    sys.path.insert(0, str(repo))
    previous = os.getcwd()
    os.chdir(str(repo))
    try:
        from core.utils.config_utils import update_key

        update_key("api.base_url", url)
        update_key("api.key", "fake")
        update_key("api.model", "fake-model")
        update_key("api.llm_support_json", False)
        update_key("whisper.language", "ja")
        update_key("whisper.detected_language", "ja")
        update_key("target_language", "简体中文")
        update_key("reflect_translate", False)
        update_key("max_workers", 1)
        update_key("burn_subtitles", False)
    finally:
        os.chdir(previous)
        sys.path.remove(str(repo))


def run(repo, code: str) -> str:
    env = {
        **os.environ,
        "ASMRDUB_PKG_PATH": ROOT,
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
            f"--- stdout ---\n{result.stdout[-5000:]}\n"
            f"--- stderr ---\n{result.stderr[-5000:]}"
        )
    return result.stdout


@pytest.fixture(scope="module")
def text_done(repo):
    run(
        repo,
        "from core import _3_1_split_nlp, _3_2_split_meaning, _4_1_summarize,"
        " _4_2_translate, _5_split_sub, _6_gen_sub, _8_1_audio_task,"
        " _8_2_dub_chunks\n"
        "_3_1_split_nlp.split_by_spacy()\n"
        "_3_2_split_meaning.split_sentences_by_meaning()\n"
        "_4_1_summarize.get_summary()\n"
        "_4_2_translate.translate_all()\n"
        "_5_split_sub.split_for_sub_main()\n"
        "_6_gen_sub.align_timestamp_main()\n"
        "_8_1_audio_task.gen_audio_task_main()\n"
        "_8_2_dub_chunks.gen_dub_chunks()\n",
    )
    return repo


def test_japanese_spacy_split_produces_sentences(text_done):
    """Requires the ja_core_news_md model; a wrong language config lands in en."""
    path = text_done / "output" / "log" / "split_by_nlp.txt"
    assert path.exists()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 3
    assert any("こんばんは" in line for line in lines)
    assert any("おやすみ" in line for line in lines)


def test_translation_reaches_every_line(text_done):
    df = pd.read_excel(str(text_done / "output" / "log" / "translation_results.xlsx"))
    assert len(df) >= 3
    assert df["Translation"].notna().all()
    assert (df["Translation"].astype(str).str.startswith("【中】")).all()


def test_source_and_translation_srt_are_written(text_done):
    for name in ("src.srt", "trans.srt"):
        path = text_done / "output" / name
        assert path.exists(), name
        assert path.read_text(encoding="utf-8").strip()


def test_srt_timestamps_come_from_the_word_alignment(text_done):
    """The first cue must start where the first word did, not at zero."""
    sys.path.insert(0, ROOT)
    from asmrdub.srt_time import parse_time

    text = (text_done / "output" / "src.srt").read_text(encoding="utf-8")
    first_block = text.strip().split("\n\n")[0].splitlines()
    start = parse_time(first_block[1].split(" --> ")[0])
    assert start == pytest.approx(WORDS[0][1], abs=0.3)


def test_task_table_has_the_columns_the_dubbing_stage_needs(text_done):
    df = pd.read_excel(str(text_done / "output" / "audio" / "tts_tasks.xlsx"))
    for column in ("number", "start_time", "end_time", "duration", "text", "origin",
                   "gap", "tolerance", "tol_dur", "est_dur", "if_too_fast", "cut_off",
                   "lines", "src_lines"):
        assert column in df.columns, f"missing {column}"
    assert len(df) >= 3


def test_task_table_keeps_the_japanese_origin(text_done):
    """`origin` is what a human reviews against; losing it makes review useless."""
    df = pd.read_excel(str(text_done / "output" / "audio" / "tts_tasks.xlsx"))
    joined = "".join(str(value) for value in df["origin"])
    assert "こんばんは" in joined


def test_task_timings_are_ordered_and_positive(text_done):
    sys.path.insert(0, ROOT)
    from asmrdub.srt_time import parse_time

    df = pd.read_excel(str(text_done / "output" / "audio" / "tts_tasks.xlsx"))
    previous_end = -1.0
    for _, row in df.iterrows():
        start = parse_time(row["start_time"])
        end = parse_time(row["end_time"])
        assert end > start, row["number"]
        assert start >= previous_end - 1e-6, f"row {row['number']} starts before the last ends"
        previous_end = end


def test_minimum_duration_does_not_swallow_short_interjections(text_done):
    """With min_subtitle_duration lowered to 1.2s, 'ふふっ' must survive.

    Upstream's 2.5s default merges it into the neighbouring line, which is
    exactly the back-and-forth texture ASMR is made of.
    """
    df = pd.read_excel(str(text_done / "output" / "audio" / "tts_tasks.xlsx"))
    assert len(df) >= 4, f"lines were merged away: {df['origin'].tolist()}"


def test_review_edit_round_trip_survives_the_dub_chunk_stage(text_done):
    """Rewriting a translation must not break chunking (upstream would raise)."""
    sys.path.insert(0, ROOT)
    from asmrdub.review_table import apply_editor_rows, to_editor_rows

    path = text_done / "output" / "audio" / "tts_tasks.xlsx"
    rows = pd.read_excel(str(path)).to_dict("records")
    edited = to_editor_rows(rows)
    edited[0]["text"] = "完全不一样的译文，和字幕文件里的内容毫无关系"
    merged = apply_editor_rows(rows, edited)
    pd.DataFrame(merged).to_excel(str(path), index=False)

    run(text_done, "from core._8_2_dub_chunks import gen_dub_chunks; gen_dub_chunks()")
    after = pd.read_excel(str(path))
    assert "完全不一样" in ast.literal_eval(str(after.loc[0, "lines"]))[0]


def test_llm_was_actually_consulted(fake_llm):
    """A cached or skipped LLM would make the whole test vacuous.

    Only two stages necessarily call out: terminology extraction and the
    faithfulness translation. The meaning-split and subtitle-split stages are
    conditional -- they only ask the LLM when a line exceeds the length limits,
    and this transcript's lines do not.
    """
    _, handler = fake_llm
    assert any("terminology consultant" in prompt for prompt in handler.prompts)
    assert any(
        "faithfully translating" in prompt or "Faithful to the original" in prompt
        for prompt in handler.prompts
    )


def test_prompts_declare_japanese_source_and_chinese_target(fake_llm):
    """A wrong config here silently produces English output."""
    _, handler = fake_llm
    translation_prompts = [
        prompt for prompt in handler.prompts
        if "Faithful to the original" in prompt or "faithfully translating" in prompt
    ]
    assert translation_prompts
    for prompt in translation_prompts:
        assert "简体中文" in prompt
        assert "ja" in prompt


def test_reflection_pass_was_skipped(fake_llm):
    """reflect_translate=false is what keeps local translation under 25 minutes."""
    _, handler = fake_llm
    assert not any(
        "optimizing the" in prompt for prompt in handler.prompts
    ), "the expressiveness pass ran despite reflect_translate=false"
