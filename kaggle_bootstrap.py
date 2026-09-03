#!/usr/bin/env python3
"""ASMR 日→中 配音 · Kaggle 引导脚本（唯一需要粘贴进 notebook 的文件）

用法
----
1. Notebook 设置：Accelerator = **GPU T4 x2**，Internet = **On**
2. 左侧 Data → Add Input，挂载：
     · 权重 Dataset（IndexTTS-2.5 / faster-whisper / GGUF，可选但强烈建议）
     · 你要转换的音频文件所在的 Dataset
3. 新建一个 cell，粘贴本文件全部内容，运行。

它只做四件事：拉代码 → 装环境 → 起服务 → 打印带密码的网址。
全部重活都在 GitHub 上的项目里，本文件保持在百行量级，方便你直接读完。

覆盖参数（可选）
--------------
    REPO   换成你自己 fork 的 overlay 仓库
    REF    换分支或 tag
    MODEL  换翻译模型（ollama 标签）
    PASSWORD 固定访问密码（留空则每次随机生成）
"""

import os
import subprocess
import sys

# ==========================================================================
REPO = "https://github.com/1432647/asmr-dub-kaggle.git"
REF = "main"
MODEL = ""          # 留空 = 用项目里钉好的默认（无审查 Gemma 4 12B）
PASSWORD = ""       # 留空 = 随机生成并打印
WORKDIR = "/kaggle/working"
# ==========================================================================

PROJECT = os.path.join(WORKDIR, "asmr-dub-kaggle")


def sh(command, **kwargs):
    print(">>> " + " ".join(command), flush=True)
    result = subprocess.run(command, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"命令失败（rc={result.returncode}）：{' '.join(command)}")


def fetch_project():
    """把 overlay 项目弄到本地。

    三种来源，按可用性依次尝试：
      1. 已挂载为 Kaggle Dataset（离线可用，认 bootstrap/bootstrap_main.py 存在）
      2. git clone（默认）
      3. 已经克隆过 → git fetch 更新
    """
    mounted = _find_mounted_project()
    if mounted:
        print(f"使用已挂载的项目：{mounted}", flush=True)
        return mounted

    if os.path.isdir(os.path.join(PROJECT, ".git")):
        sh(["git", "-C", PROJECT, "fetch", "--depth", "1", "origin", REF])
        sh(["git", "-C", PROJECT, "checkout", "-q", "--force", "FETCH_HEAD"])
    else:
        sh(["git", "clone", "--depth", "1", "--branch", REF, REPO, PROJECT])
    return PROJECT


def _find_mounted_project(root="/kaggle/input", max_depth=4):
    if not os.path.isdir(root):
        return None
    base_depth = root.rstrip("/").count("/")
    for current, subdirs, _ in os.walk(root):
        if current.count("/") - base_depth >= max_depth:
            subdirs[:] = []
            continue
        if os.path.isfile(os.path.join(current, "bootstrap", "bootstrap_main.py")):
            return current
    return None


def main():
    project = fetch_project()

    # 挂载点是只读的：把项目复制到 working，setup 需要写入。
    if not project.startswith(WORKDIR):
        import shutil

        if os.path.isdir(PROJECT):
            shutil.rmtree(PROJECT)
        shutil.copytree(project, PROJECT)
        project = PROJECT

    bootstrap = [sys.executable, os.path.join(project, "bootstrap", "bootstrap_main.py")]
    if MODEL:
        bootstrap += ["--model", MODEL]
    if PASSWORD:
        bootstrap += ["--password", PASSWORD]
    sh(bootstrap)

    state = os.path.join(WORKDIR, "asmrdub_state.json")
    # 前台运行：keepalive 会持续打印各服务状态，网址和密码在上方横幅里。
    # 停止本 cell 不会杀掉服务（它们都在独立 session 中）。
    sh([sys.executable, os.path.join(project, "runtime", "run_all.py"),
        "--state", state])


if __name__ == "__main__":
    main()
