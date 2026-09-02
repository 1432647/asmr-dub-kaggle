"""Streamlit UI for the ASMR JP->ZH dubbing pipeline.

Replaces VideoLingo's `st.py`, which is built around downloading a YouTube
video, burning subtitles and previewing an mp4 -- none of which applies here.

Four things this UI must do that upstream's does not:

1. Gate access. The cloudflared quick tunnel URL is public; anyone holding it
   can drive the GPUs. A password is the only barrier.
2. Take input from a mounted Kaggle Dataset. A 15-minute file uploaded through
   the tunnel frequently dies halfway.
3. Show logs. Upstream's rich output goes to the notebook's stdout, which is
   invisible from the browser. stdout is teed to a file and tailed here.
4. Let a human fix the transcript before TTS. Whispered Japanese is the weakest
   link in the chain, and 40 minutes of TTS on a bad transcript is wasted.
"""

from __future__ import annotations

import glob
import io
import os
import shutil
import sys
import time
import zipfile

import streamlit as st

# stdout must be teed before any VideoLingo import: its modules print at import
# time and those lines belong in the log panel too.
LOG_PATH = os.environ.get("ASMRDUB_LOG", "output/pipeline.log")


class _Tee:
    """Duplicate a stream to a file, so the browser and the notebook agree."""

    def __init__(self, stream, path):
        self.stream = stream
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.file = open(path, "a", encoding="utf-8", errors="replace")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


if not isinstance(sys.stdout, _Tee):
    sys.stdout = _Tee(sys.stdout, LOG_PATH)
    sys.stderr = _Tee(sys.stderr, LOG_PATH)

import pandas as pd  # noqa: E402

from core import (  # noqa: E402
    _2_asr,
    _3_1_split_nlp,
    _3_2_split_meaning,
    _4_1_summarize,
    _4_2_translate,
    _5_split_sub,
    _6_gen_sub,
    _8_1_audio_task,
    _8_2_dub_chunks,
    _9_refer_audio,
    _10_gen_audio,
    _11_merge_audio,
)
from core._1_ytdlp import write_input_manifest  # noqa: E402
from core.asr_backend import worker_client  # noqa: E402
from core.st_utils.task_runner import TaskRunner  # noqa: E402
from core.utils import load_key, update_key  # noqa: E402
from core.utils.models import _8_1_AUDIO_TASK, _OUTPUT_DIR  # noqa: E402

sys.path.insert(0, os.environ.get("ASMRDUB_PKG_PATH", ""))
from asmrdub.review_table import (  # noqa: E402
    apply_editor_rows,
    to_editor_rows,
    validate_editor_rows,
)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma")
TEXT_DONE = os.path.join(_OUTPUT_DIR, ".subtitle_done")
REVIEW_DONE = os.path.join(_OUTPUT_DIR, ".review_done")
AUDIO_DONE = os.path.join(_OUTPUT_DIR, ".dubbing_done")
DUB_WAV = os.path.join(_OUTPUT_DIR, "dub_44k_stereo.wav")

st.set_page_config(page_title="ASMR 日→中 配音", page_icon="🎧", layout="wide")


# --------------------------------------------------------------------------
# Access gate
# --------------------------------------------------------------------------


def require_password() -> bool:
    expected = os.environ.get("ASMRDUB_PASSWORD", "")
    if not expected:
        st.error("未设置 ASMRDUB_PASSWORD，拒绝在公网隧道上无鉴权运行。")
        return False
    if st.session_state.get("authed"):
        return True
    st.title("🎧 ASMR 日→中 配音")
    st.caption("此地址通过 Cloudflare 隧道公开可访问，需要密码。")
    with st.form("login"):
        entered = st.text_input("访问密码", type="password")
        if st.form_submit_button("进入") :
            if entered == expected:
                st.session_state["authed"] = True
                st.rerun()
            else:
                # Constant-ish delay: this is a single-user tool behind a random
                # 16-char password, so rate limiting is enough.
                time.sleep(1.0)
                st.error("密码错误")
    return False


# --------------------------------------------------------------------------
# Shared widgets
# --------------------------------------------------------------------------


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w", encoding="utf-8").close()


def _clear(path: str) -> None:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def log_panel() -> None:
    with st.expander("📜 运行日志", expanded=True):
        lines = st.slider("显示末尾行数", 20, 400, 80, step=20, key="log_lines")
        _log_body(lines)


@st.fragment(run_every=2)
def _log_body(lines: int) -> None:
    if not os.path.exists(LOG_PATH):
        st.info("暂无日志。")
        return
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
        tail = fh.readlines()[-lines:]
    st.code("".join(tail) or "(空)", language="log")


@st.fragment(run_every=1)
def task_controls(runner_key: str) -> None:
    """Progress bar plus pause/stop, refreshing while a stage runs."""
    runner = TaskRunner.get(st.session_state, runner_key)
    if runner.state == "idle":
        return
    label = (
        f"({runner.current_step + 1}/{runner.total_steps}) {runner.current_label}"
        if runner.current_step >= 0
        else ""
    )
    if runner.is_active:
        (st.warning if runner.state == "paused" else st.info)(
            f"{'⏸️ 已暂停' if runner.state == 'paused' else '⏳ 正在运行'} {label}"
        )
        st.progress(runner.progress)
        left, right = st.columns(2)
        with left:
            if runner.state == "paused":
                if st.button("▶️ 继续", key=f"{runner_key}_resume", use_container_width=True):
                    runner.resume()
                    st.rerun()
            elif st.button("⏸️ 暂停", key=f"{runner_key}_pause", use_container_width=True):
                runner.pause()
                st.rerun()
        with right:
            if st.button("⏹️ 停止", key=f"{runner_key}_stop", use_container_width=True,
                         type="primary"):
                runner.stop()
                st.rerun()
    elif runner.state == "completed":
        st.success("阶段完成")
        runner.reset()
        st.rerun(scope="app")
    elif runner.state == "stopped":
        st.warning(f"⏹️ 已停止 {label}")
        if st.button("确定", key=f"{runner_key}_ack_stop"):
            runner.reset()
            st.rerun(scope="app")
    elif runner.state == "error":
        st.error(f"❌ 出错：{runner.error_msg}")
        st.caption("展开上方日志查看完整堆栈。中间产物已保存，修好后可从同一阶段续跑。")
        if st.button("确定", key=f"{runner_key}_ack_error"):
            runner.reset()
            st.rerun(scope="app")


# --------------------------------------------------------------------------
# Section 1: input
# --------------------------------------------------------------------------


def dataset_audio_files() -> list[str]:
    """Audio under /kaggle/input, newest first.

    Mounted datasets are read-only, which is exactly what we want for the
    source material: nothing in the pipeline can damage the original.
    """
    root = os.environ.get("ASMRDUB_INPUT_ROOT", "/kaggle/input")
    if not os.path.isdir(root):
        return []
    found = [
        path
        for path in glob.glob(os.path.join(root, "**", "*"), recursive=True)
        if path.lower().endswith(AUDIO_EXTENSIONS) and os.path.isfile(path)
    ]
    return sorted(found, key=lambda p: (-os.path.getmtime(p), p))


def input_section() -> str | None:
    st.header("① 选择音频")
    current = _current_input()
    if current:
        st.success(f"已选：`{current}`")
        size = os.path.getsize(current) / 1e6
        st.caption(f"{size:.1f} MB")
        if st.button("🗑️ 换一个音频（清空所有中间产物）"):
            _reset_output()
            st.rerun()
        return current

    candidates = dataset_audio_files()
    if candidates:
        choice = st.selectbox(
            "从已挂载的 Kaggle Dataset 中选择",
            options=["（不选）"] + candidates,
            format_func=lambda p: p if p == "（不选）" else os.path.relpath(
                p, os.environ.get("ASMRDUB_INPUT_ROOT", "/kaggle/input")
            ),
        )
        if choice != "（不选）" and st.button("✅ 使用这个文件", type="primary"):
            _adopt(choice, copy=True)
            st.rerun()
    else:
        st.info(
            "未在 /kaggle/input 找到音频。左侧 Data 面板 → Add Input → 挂载含音频的 "
            "Dataset，或用下方上传（长文件经隧道上传容易断）。"
        )

    uploaded = st.file_uploader(
        "或直接上传", type=[ext.lstrip(".") for ext in AUDIO_EXTENSIONS]
    )
    if uploaded is not None:
        target = os.path.join(_OUTPUT_DIR, uploaded.name)
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(uploaded.getbuffer())
        _adopt(target, copy=False)
        st.rerun()
    return None


def _adopt(path: str, copy: bool) -> str:
    """Register a file as the pipeline input.

    Dataset files are copied into output/ because `/kaggle/input` is read-only
    and several upstream helpers expect to write next to the media file.
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    if copy:
        target = os.path.join(_OUTPUT_DIR, os.path.basename(path))
        if os.path.abspath(target) != os.path.abspath(path):
            shutil.copy2(path, target)
        path = target
    write_input_manifest(path.replace("\\", "/"), "audio", _OUTPUT_DIR)
    return path


def _current_input() -> str | None:
    from core._1_ytdlp import _read_input_manifest

    found = _read_input_manifest(_OUTPUT_DIR)
    if not found:
        return None
    path, _ = found
    return path if os.path.exists(path) else None


def _reset_output() -> None:
    """Wipe intermediates so the next run starts clean.

    Deliberately does not touch /kaggle/input.
    """
    if os.path.isdir(_OUTPUT_DIR):
        shutil.rmtree(_OUTPUT_DIR, ignore_errors=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Section 2: transcript + translation
# --------------------------------------------------------------------------


def text_steps() -> list[tuple[str, callable]]:
    return [
        ("语音识别 + 人声分离", _2_asr.transcribe),
        (
            "断句（spaCy + LLM）",
            lambda: (
                _3_1_split_nlp.split_by_spacy(),
                _3_2_split_meaning.split_sentences_by_meaning(),
            ),
        ),
        (
            "术语提取 + 翻译",
            lambda: (_4_1_summarize.get_summary(), _4_2_translate.translate_all()),
        ),
        (
            "字幕切分与时间轴对齐",
            lambda: (
                _5_split_sub.split_for_sub_main(),
                _6_gen_sub.align_timestamp_main(),
            ),
        ),
        (
            "生成配音任务",
            lambda: (
                _8_1_audio_task.gen_audio_task_main(),
                _8_2_dub_chunks.gen_dub_chunks(),
            ),
        ),
        ("标记完成", lambda: _touch(TEXT_DONE)),
    ]


def text_section() -> bool:
    st.header("② 识别与翻译")
    runner = TaskRunner.get(st.session_state, "_text_runner")
    done = os.path.exists(TEXT_DONE) and os.path.exists(_8_1_AUDIO_TASK)

    with st.container(border=True):
        st.markdown(
            "1. 语音识别 + 人声分离 → 2. 断句 → 3. 翻译 → "
            "4. 时间轴对齐 → 5. 生成配音任务"
        )
        if done:
            st.success("已完成。")
            return True
        if runner.state != "idle":
            task_controls("_text_runner")
            return False
        if not _worker_ready_banner():
            return False
        if st.button("▶️ 开始识别与翻译", type="primary", key="start_text"):
            _clear(TEXT_DONE)
            runner.start(text_steps())
            st.rerun()
    return False


def _worker_ready_banner() -> bool:
    status = worker_client.health()
    if status is None:
        st.error(
            "GPU worker 未就绪（127.0.0.1:7861 无响应）。请回到 notebook 查看 "
            "worker 日志——多半是权重路径不对或显存不足。"
        )
        return False
    vram = status.get("vram") or {}
    st.caption(
        "GPU worker 就绪 · 当前载入：%s · 显存 %s/%s GB"
        % (
            status.get("loaded") or "无",
            vram.get("free_gb", "?"),
            vram.get("total_gb", "?"),
        )
    )
    return True


# --------------------------------------------------------------------------
# Section 3: human review
# --------------------------------------------------------------------------


def review_section() -> bool:
    st.header("③ 人工复核（可跳过）")
    if os.path.exists(REVIEW_DONE):
        st.success("已确认。")
        if st.button("↩️ 重新复核", key="reopen_review"):
            _clear(REVIEW_DONE)
            st.rerun()
        return True

    with st.container(border=True):
        st.caption(
            "日语 ASMR 多耳语和气声，识别是整条链最弱的一环。这里改译文最省时间——"
            "改完再合成，比合成 40 分钟后重做便宜得多。时间轴由对齐得出，不可编辑。"
        )
        rows = pd.read_excel(_8_1_AUDIO_TASK).to_dict("records")
        editable = pd.DataFrame(to_editor_rows(rows))
        edited = st.data_editor(
            editable,
            use_container_width=True,
            height=460,
            num_rows="fixed",
            disabled=["number", "start", "end", "duration"],
            column_config={
                "number": st.column_config.NumberColumn("序号", width="small"),
                "start": st.column_config.TextColumn("起", width="small"),
                "end": st.column_config.TextColumn("止", width="small"),
                "duration": st.column_config.NumberColumn(
                    "时长", format="%.2f", width="small"
                ),
                "origin": st.column_config.TextColumn("日文（识别）", width="large"),
                "text": st.column_config.TextColumn("中文（译文）", width="large"),
            },
            key="review_editor",
        )
        left, right = st.columns(2)
        with left:
            if st.button("💾 保存修改并确认", type="primary", key="save_review"):
                edits = edited.to_dict("records")
                problems = validate_editor_rows(edits)
                if problems:
                    st.error("请先修正：\n\n" + "\n".join(f"- {p}" for p in problems))
                else:
                    merged = apply_editor_rows(rows, edits)
                    pd.DataFrame(merged).to_excel(_8_1_AUDIO_TASK, index=False)
                    _touch(REVIEW_DONE)
                    st.rerun()
        with right:
            if st.button("⏭️ 不改，直接合成", key="skip_review"):
                _touch(REVIEW_DONE)
                st.rerun()
    return False


# --------------------------------------------------------------------------
# Section 4: dubbing
# --------------------------------------------------------------------------


def audio_steps() -> list[tuple[str, callable]]:
    return [
        ("提取参考音频", _9_refer_audio.extract_refer_audio_main),
        ("逐句合成中文配音", _10_gen_audio.gen_audio),
        ("混音（环境音 + 配音，44.1k 立体声）", _11_merge_audio.merge_full_audio),
        ("标记完成", lambda: _touch(AUDIO_DONE)),
    ]


def audio_section() -> None:
    st.header("④ 合成配音")
    runner = TaskRunner.get(st.session_state, "_audio_runner")
    done = os.path.exists(AUDIO_DONE) and os.path.exists(DUB_WAV)

    with st.container(border=True):
        if done:
            st.success("配音完成。")
            st.audio(DUB_WAV)
            _download_buttons()
            if st.button("🔁 重新合成（保留识别与翻译）", key="redo_audio"):
                _clear(AUDIO_DONE)
                for pattern in ("output/audio/segs/*", "output/audio/tmp/*"):
                    for path in glob.glob(pattern):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                st.rerun()
            return
        if runner.state != "idle":
            task_controls("_audio_runner")
            return
        if not _worker_ready_banner():
            return
        st.caption("每句用原片同一句的声音克隆，音色与声像跟随原声优。")
        if st.button("▶️ 开始合成", type="primary", key="start_audio"):
            _clear(AUDIO_DONE)
            runner.start(audio_steps())
            st.rerun()


def _download_buttons() -> None:
    columns = st.columns(3)
    for column, path, label in (
        (columns[0], DUB_WAV, "⬇️ WAV（44.1k 立体声）"),
        (columns[1], os.path.join(_OUTPUT_DIR, "dub.mp3"), "⬇️ MP3"),
    ):
        if os.path.exists(path):
            with column, open(path, "rb") as fh:
                st.download_button(label, fh.read(), file_name=os.path.basename(path))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in os.listdir(_OUTPUT_DIR):
            if name.endswith((".srt", ".json", ".xlsx")):
                archive.write(os.path.join(_OUTPUT_DIR, name), name)
    buffer.seek(0)
    with columns[2]:
        st.download_button("⬇️ 字幕与日志包", buffer, file_name="asmr_dub_meta.zip")


# --------------------------------------------------------------------------
# Sidebar + main
# --------------------------------------------------------------------------


def sidebar() -> None:
    with st.sidebar:
        st.subheader("设置")
        st.caption(f"翻译模型：`{load_key('api.model')}`")
        st.caption(f"目标语言：{load_key('target_language')}")

        reflect = st.checkbox(
            "翻译二次润色（慢一倍）",
            value=bool(load_key("reflect_translate")),
            help="开启后每段文本额外跑一次 LLM 反思重写。本地 12B 上会把翻译时间翻倍。",
        )
        if reflect != bool(load_key("reflect_translate")):
            update_key("reflect_translate", reflect)
            st.rerun()

        speed_max = st.slider(
            "配音最大加速倍率",
            1.0, 1.5, float(load_key("speed_factor.max")), step=0.05,
            help="译文塞不进原时间轴时的变速上限。ASMR 加速过头会失真。",
        )
        if abs(speed_max - float(load_key("speed_factor.max"))) > 1e-6:
            update_key("speed_factor.max", speed_max)
            update_key("speed_factor.accept", min(speed_max, speed_max - 0.05 or 1.0))
            st.rerun()

        st.divider()
        if st.button("🧹 释放 GPU 显存"):
            worker_client.unload()
            st.toast("已请求 worker 卸载模型")
        st.caption(
            "中间产物在 output/，每阶段可续跑；会话被回收后重新运行引导 cell 即可。"
        )


def main() -> None:
    if not require_password():
        return
    st.title("🎧 ASMR 日→中 配音")
    sidebar()
    log_panel()

    if not input_section():
        return
    st.divider()
    if not text_section():
        return
    st.divider()
    if not review_section():
        return
    st.divider()
    audio_section()


main()
