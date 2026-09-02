"""asmrdub -- pure-logic core for the ASMR JP->ZH dubbing pipeline.

Nothing in this package imports torch, streamlit or VideoLingo, so every
module here runs (and is tested) on a laptop. The GPU-bound and
VideoLingo-bound code lives in ``worker/`` and ``overlay/``.
"""

__all__ = [
    "asr_format",
    "chunking",
    "mixdown",
    "patches",
    "pins",
    "refer_window",
    "review_table",
    "srt_time",
    "vl_config",
]
