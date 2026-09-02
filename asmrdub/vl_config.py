"""Generate the VideoLingo config.yaml overrides for this pipeline.

Upstream defaults assume a hosted frontier model doing English->Chinese on
talking-head video. Almost every value below is wrong for local Gemma 4 on
Japanese ASMR audio, and the comments say why.
"""

from __future__ import annotations

from asmrdub import pins


def config_overrides(
    ollama_model: str,
    llm_base_url: str,
    target_language: str = "简体中文",
    max_workers: int = 2,
    reflect_translate: bool = False,
    model_cache_dir: str = "./_model_cache",
) -> dict:
    """Build the nested dict of keys to merge into VideoLingo's config.yaml.

    Only keys that already exist upstream are returned: ``update_key`` raises
    KeyError on unknown keys, so inventing one would break setup.
    """
    return {
        # ---- LLM: point at local ollama -------------------------------
        # ask_gpt() refuses to run on an empty key even though ollama ignores
        # it. base_url already ends in /v1 so ask_gpt won't append another.
        "api": {
            "key": "ollama",
            "base_url": llm_base_url,
            "model": ollama_model,
            # Gemma has no JSON-mode; ask_gpt falls back to json_repair on the
            # ```json fences the prompts request, which works fine.
            "llm_support_json": False,
        },
        # 12B on one T4 fits ~2 concurrent contexts; more just thrashes.
        "max_workers": max_workers,
        # ---- Languages -----------------------------------------------
        "target_language": target_language,
        "whisper": {
            "model": "large-v3",
            "language": pins.LANG_JA,
            "detected_language": pins.LANG_JA,
            "runtime": "local",
        },
        # ---- Separation ----------------------------------------------
        # We run our own full-rate HDemucs pass and hand back both stems, so
        # upstream must not re-run its 16k mono version.
        "demucs": True,
        # ---- Video: there is none ------------------------------------
        "burn_subtitles": False,
        "ffmpeg_gpu": False,
        # ---- Translation cost ----------------------------------------
        # reflect_translate=true runs a second "reflect and rewrite" LLM pass
        # over every chunk. On a local 12B that roughly doubles a 20-minute
        # stage for a modest gain; off by default here.
        "reflect_translate": reflect_translate,
        "pause_before_translate": False,
        # Japanese has no spaces, so max_split_length counts spaCy tokens.
        # 20 tokens of Japanese is a long subtitle; 14 keeps lines dubbable.
        "max_split_length": 14,
        "subtitle": {
            # Chinese counts 1.75 per char in calc_len; 60 ~ 34 Chinese chars.
            "max_length": 60,
            "target_multiplier": 1.2,
        },
        # Local model, short context: keep the summary prompt small.
        "summary_length": 3000,
        # ---- Dubbing timing -----------------------------------------
        "tts_method": "custom_tts",
        "speed_factor": {
            # ASMR is slow and intimate. Speeding a whisper up 40% turns it
            # into a chipmunk, so the ceiling is much tighter than upstream's.
            "min": 1.0,
            "accept": 1.10,
            "max": 1.20,
        },
        # Upstream force-extends every sub to 2.5s, which merges the short
        # back-and-forth lines ASMR is made of. 1.2s preserves them.
        "min_subtitle_duration": 1.2,
        "min_trim_duration": 2.5,
        # Silence between lines can be absorbed to fit a long dub; ASMR has
        # long deliberate pauses, so allow a generous steal.
        "tolerance": 1.8,
        "model_dir": model_cache_dir,
    }


def flatten(overrides: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten nested overrides into VideoLingo's dotted update_key paths."""
    items: list[tuple[str, object]] = []
    for key, value in overrides.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            items.extend(flatten(value, prefix=f"{path}."))
        else:
            items.append((path, value))
    return items
