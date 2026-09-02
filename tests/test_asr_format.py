from types import SimpleNamespace

from asmrdub.asr_format import (
    merge_results,
    repair_word_times,
    to_videolingo,
    words_from_segment,
)


def _word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end)


def _segment(text, start, end, words=None):
    return SimpleNamespace(text=text, start=start, end=end, words=words)


def test_converts_faster_whisper_objects():
    segments = [_segment("こんにちは", 0.0, 1.5, [_word("こんにちは", 0.0, 1.5)])]
    result = to_videolingo(segments, "ja")
    assert result["language"] == "ja"
    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["text"] == "こんにちは"
    assert seg["words"][0]["word"] == "こんにちは"
    assert seg["words"][0]["start"] == 0.0


def test_accepts_dicts_too():
    segments = [
        {"text": "はい", "start": 1.0, "end": 2.0,
         "words": [{"word": "はい", "start": 1.0, "end": 2.0}]}
    ]
    assert to_videolingo(segments, "ja")["segments"][0]["text"] == "はい"


def test_synthesises_a_word_when_none_present():
    """word_timestamps=False must not break sentence alignment downstream."""
    words = words_from_segment(_segment("ふぅ", 3.0, 4.0, None))
    assert words == [{"word": "ふぅ", "start": 3.0, "end": 4.0}]


def test_empty_segment_is_dropped():
    result = to_videolingo([_segment("   ", 0.0, 1.0, [])], "ja")
    assert result["segments"] == []


def test_blank_words_are_skipped():
    words = words_from_segment(
        _segment("ねえ", 0.0, 1.0, [_word(" ", 0.0, 0.1), _word("ねえ", 0.1, 1.0)])
    )
    assert [w["word"] for w in words] == ["ねえ"]


def test_missing_timings_are_filled_forward():
    words = [
        {"word": "a", "start": 0.0, "end": 0.5},
        {"word": "b", "start": None, "end": None},
        {"word": "c", "start": 0.9, "end": 1.4},
    ]
    repaired = repair_word_times(words, 0.0, 1.4)
    assert repaired[1]["start"] == 0.5
    assert repaired[1]["end"] == 0.5
    assert all(w["end"] >= w["start"] for w in repaired)


def test_inverted_timings_are_clamped_monotonic():
    words = [
        {"word": "a", "start": 1.0, "end": 2.0},
        {"word": "b", "start": 0.5, "end": 0.2},
    ]
    repaired = repair_word_times(words, 1.0, 2.0)
    assert repaired[1]["start"] >= repaired[0]["end"]
    assert repaired[1]["end"] >= repaired[1]["start"]


def test_time_offset_shifts_everything():
    segments = [_segment("あ", 0.0, 1.0, [_word("あ", 0.0, 1.0)])]
    result = to_videolingo(segments, "ja", time_offset=100.0)
    seg = result["segments"][0]
    assert seg["start"] == 100.0
    assert seg["end"] == 101.0
    assert seg["words"][0]["start"] == 100.0


def test_text_falls_back_to_joined_words():
    segments = [_segment("", 0.0, 1.0, [_word("あ", 0.0, 0.5), _word("い", 0.5, 1.0)])]
    assert to_videolingo(segments, "ja")["segments"][0]["text"] == "あい"


def test_merge_sorts_chronologically():
    first = to_videolingo([_segment("b", 10.0, 11.0, [_word("b", 10.0, 11.0)])], "ja")
    second = to_videolingo([_segment("a", 0.0, 1.0, [_word("a", 0.0, 1.0)])], "ja")
    merged = merge_results([first, second])
    assert [s["text"] for s in merged["segments"]] == ["a", "b"]
    assert merged["language"] == "ja"


def test_merge_of_nothing_still_reports_a_language():
    assert merge_results([])["language"] == "ja"
